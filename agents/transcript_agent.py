"""
TranscriptAgent
---------------
Responsibility: Given a YouTube URL, return a plain-text transcript.

Strategy (hybrid — fastest reliable path first):
  1. youtube-transcript-api  — fetches auto-generated captions directly.
     No download, no ffmpeg, no bot detection risk. Works on ~90% of videos.
  2. yt-dlp + Groq Whisper   — downloads audio and transcribes it.
     Used only when captions are unavailable.

Agentic features (audio-download path):
- Classifies errors before deciding next action
- Retries with backoff on rate limits and network errors
- Stops immediately on unrecoverable errors (bot detected, private video)
- Tries multiple yt-dlp player clients for reliability
- Supports YouTube cookies via YOUTUBE_COOKIES env var
"""
import os
import re
import time
import shutil
import tempfile
import yt_dlp
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Error classification ──────────────────────────────────────────────────────

def _classify(error_msg: str) -> str:
    msg = error_msg.lower()
    if "sign in" in msg or "confirm you're not a bot" in msg:
        return "bot_detected"
    if "requested format" in msg or "no video formats" in msg:
        return "format_error"
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "rate_limit"
    if any(s in msg for s in ["video unavailable", "private video",
                               "this video is not available",
                               "video has been removed"]):
        return "video_unavailable"
    if "ffmpeg" in msg or "ffprobe" in msg:
        return "ffmpeg_error"
    if any(s in msg for s in ["timed out", "connection", "network"]):
        return "network_error"
    return "unknown"

UNRECOVERABLE = {"bot_detected", "video_unavailable", "ffmpeg_error"}
BACKOFF_RETRY  = {"rate_limit", "network_error"}

USER_MESSAGES = {
    "bot_detected":      "YouTube blocked this as a bot. Set YOUTUBE_COOKIES in Railway Variables with fresh cookies from Chrome.",
    "format_error":      "No compatible audio format found for this video.",
    "rate_limit":        "Rate limited. Try again in a few minutes.",
    "video_unavailable": "This video is private, deleted, or region-restricted.",
    "ffmpeg_error":      "Audio processing failed — ffmpeg is missing or broken.",
    "network_error":     "Network error. Check connection and try again.",
    "unknown":           "An unexpected error occurred.",
}


class TranscriptAgent:
    """
    Agent 1 — Returns a transcript for a YouTube URL.

    Primary path : youtube-transcript-api (captions, instant, no bot risk).
    Fallback path: yt-dlp audio download + Groq Whisper transcription.

    After run() returns, self.last_source is set to:
        "captions_api"   — transcript came from auto-generated captions
        "audio_download" — transcript came from Whisper on downloaded audio

    Usage:
        agent = TranscriptAgent()
        transcript = agent.run("https://www.youtube.com/watch?v=...")
        print(agent.last_source)
    """

    MAX_FILE_SIZE_MB = 24
    MAX_RETRIES      = 3

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )
        self.client              = Groq(api_key=api_key)
        self._cookies_temp_path  = None
        self.last_source         = None
        print("[TranscriptAgent] Initialised")

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, youtube_url: str) -> str:
        self._validate_url(youtube_url)
        print(f"[TranscriptAgent] Processing: {youtube_url}")

        # ── Primary: captions API (fast, no bot detection) ────────────────
        transcript = self._get_transcript_via_api(youtube_url)
        if transcript:
            self.last_source = "captions_api"
            print(f"[TranscriptAgent] Transcript fetched via captions API (instant) "
                  f"— {len(transcript)} characters")
            return transcript

        # ── Fallback: audio download + Whisper ────────────────────────────
        print("[TranscriptAgent] No captions found — downloading audio...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = self._download_with_retry(youtube_url, tmp_dir)
            transcript = self._transcribe_with_retry(audio_path)

        self.last_source = "audio_download"
        print(f"[TranscriptAgent] Done — {len(transcript)} characters")
        return transcript

    # ── Captions API (primary path) ───────────────────────────────────────

    def _get_transcript_via_api(self, url: str):
        """
        Fetch captions via youtube-transcript-api (v1.x instance-based API).
        Returns a non-empty string on success, None on any failure
        (no captions, disabled, wrong language, package missing, etc.).
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return None

        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not match:
            return None
        video_id = match.group(1)

        try:
            ytt      = YouTubeTranscriptApi()
            segments = ytt.fetch(video_id, languages=["en", "en-US", "en-GB"])
            text     = " ".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception:
            return None

    # ── Download with retry ───────────────────────────────────────────────────

    def _download_with_retry(self, url: str, output_dir: str) -> str:
        print("[TranscriptAgent] Step 1/2 — Downloading audio ...")

        last_error = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self._download_audio(url, output_dir)
            except RuntimeError as e:
                last_error    = str(e)
                error_type    = _classify(last_error)
                print(f"[TranscriptAgent] Download error ({error_type}): {last_error[:80]}")

                if error_type in UNRECOVERABLE:
                    raise RuntimeError(USER_MESSAGES.get(error_type, last_error))

                if error_type in BACKOFF_RETRY and attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"[TranscriptAgent] Backing off {wait}s before retry")
                    time.sleep(wait)
                    continue

                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        USER_MESSAGES.get(error_type, last_error)
                    )

                # format_error / unknown on a non-final attempt: small fixed pause
                time.sleep(1)

        raise RuntimeError(f"Download failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _download_audio(self, url: str, output_dir: str) -> str:
        output_template = os.path.join(output_dir, "audio.%(ext)s")

        # Build yt-dlp options
        node_path   = shutil.which("node") or "/usr/bin/node"
        js_runtimes = {"node": {"path": node_path}} if os.path.exists(node_path) else {}

        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
            "quiet":       True,
            "no_warnings": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "ios", "tv_embedded", "web"],
                }
            },
            "js_runtimes": js_runtimes,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
            },
        }

        # Add cookies if set
        cookies_path = self._get_cookies_path()
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
            print("[TranscriptAgent] Using cookies for authentication")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise RuntimeError(str(e))
        finally:
            if self._cookies_temp_path:
                try:
                    os.unlink(self._cookies_temp_path)
                except OSError:
                    pass
                self._cookies_temp_path = None

        audio_path = self._find_audio_file(output_dir)
        size_mb    = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"[TranscriptAgent] Audio downloaded — {size_mb:.1f} MB")

        if size_mb > self.MAX_FILE_SIZE_MB:
            raise RuntimeError(
                f"Audio file is {size_mb:.1f} MB, exceeds Groq Whisper's "
                f"25 MB limit. Try a shorter video."
            )
        return audio_path

    def _find_audio_file(self, directory: str) -> str:
        audio_extensions = {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".webm"}
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in audio_extensions:
                return os.path.join(directory, filename)
        raise RuntimeError(
            "No audio file found after download. Check that ffmpeg is installed."
        )

    def _get_cookies_path(self):
        content = os.environ.get("YOUTUBE_COOKIES", "").strip()
        if not content:
            return None
        # If it's a file path that exists on disk, use it directly
        if os.path.exists(content):
            return content
        # Treat any non-path value as raw Netscape cookie content.
        # A valid cookies.txt file contains tab-separated fields; any
        # multi-line value with a tab character is almost certainly cookie
        # content regardless of what character the first line starts with.
        # We also accept single-line values that are not a resolvable path,
        # so the user is never silently left without cookies.
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            )
            tmp.write(content)
            tmp.flush()
            tmp.close()
            self._cookies_temp_path = tmp.name
            return tmp.name
        except Exception:
            return None

    # ── Transcribe with retry ─────────────────────────────────────────────────

    def _transcribe_with_retry(self, audio_path: str) -> str:
        print("[TranscriptAgent] Step 2/2 — Transcribing via Groq Whisper ...")

        last_error = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                with open(audio_path, "rb") as f:
                    response = self.client.audio.transcriptions.create(
                        file=f,
                        model="whisper-large-v3-turbo",
                        response_format="text",
                        temperature=0.0,
                    )
                transcript = str(response).strip()
                if not transcript:
                    raise RuntimeError("Whisper returned an empty transcript.")
                return transcript

            except Exception as e:
                last_error = str(e)
                error_type = _classify(last_error)
                print(f"[TranscriptAgent] Transcription error ({error_type}): attempt {attempt}")

                if error_type in BACKOFF_RETRY and attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"[TranscriptAgent] Backing off {wait}s")
                    time.sleep(wait)
                    continue

                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(f"Transcription failed: {last_error}")

        raise RuntimeError(f"Transcription failed after {self.MAX_RETRIES} attempts")

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_url(self, url: str) -> None:
        if not url or not isinstance(url, str):
            raise ValueError("YouTube URL must be a non-empty string.")
        url = url.strip()
        if "youtube.com" not in url and "youtu.be" not in url:
            raise ValueError(f"'{url}' does not look like a YouTube URL.")


if __name__ == "__main__":
    import sys
    TEST_URL = "https://www.youtube.com/watch?v=arj7oStGLkU"
    print("=" * 60)
    print("TranscriptAgent — standalone test")
    print("=" * 60)
    try:
        agent      = TranscriptAgent()
        transcript = agent.run(TEST_URL)
        print("\n--- TRANSCRIPT PREVIEW (first 500 chars) ---")
        print(transcript[:500])
        print(f"\nTotal   : {len(transcript)} characters")
        print(f"Source  : {agent.last_source}")
        print("\nTranscriptAgent test PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        sys.exit(1)
