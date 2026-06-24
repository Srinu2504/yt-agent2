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
        print("[TranscriptAgent] Validating URL...")
        self._validate_url(youtube_url)
        print(f"[TranscriptAgent] Processing: {youtube_url}")

        # ── Primary: captions API (fast, no bot detection) ────────────────
        print("[TranscriptAgent] Trying captions API first (instant, no download needed)...")
        transcript = self._get_transcript_via_api(youtube_url)
        if transcript:
            self.last_source = "captions_api"
            print(f"[TranscriptAgent] ✅ Captions fetched via API — {len(transcript)} characters")
            return transcript

        # ── Fallback: audio download + Whisper ────────────────────────────
        print("[TranscriptAgent] No captions available — falling back to audio download...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = self._download_with_retry(youtube_url, tmp_dir)
            transcript = self._transcribe_with_retry(audio_path)

        self.last_source = "audio_download"
        print(f"[TranscriptAgent] Pipeline complete — {len(transcript)} characters transcribed")
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
            from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
        except ImportError:
            return None

        print("[TranscriptAgent] Extracting video ID from URL...")
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
        if not match:
            return None
        video_id = match.group(1)
        print(f"[TranscriptAgent] Video ID: {video_id}")

        print("[TranscriptAgent] Fetching captions from YouTube...")
        try:
            ytt      = YouTubeTranscriptApi()
            segments = list(ytt.fetch(video_id, languages=["en", "en-US", "en-GB"]))
            print(f"[TranscriptAgent] ✅ Captions found — {len(segments)} segments")
            text = " ".join(seg.text for seg in segments).strip()
            return text if text else None
        except TranscriptsDisabled:
            print("[TranscriptAgent] No captions available for this video")
            return None
        except NoTranscriptFound:
            print("[TranscriptAgent] No transcript found for this video")
            return None
        except Exception as e:
            print(f"[TranscriptAgent] Caption API error: {e}")
            return None

    # ── Download with retry ───────────────────────────────────────────────────

    def _download_with_retry(self, url: str, output_dir: str) -> str:
        last_error = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"[TranscriptAgent] Starting audio download (attempt {attempt} of {self.MAX_RETRIES})...")
            try:
                return self._download_audio(url, output_dir)
            except RuntimeError as e:
                last_error = str(e)
                error_type = _classify(last_error)
                print(f"[TranscriptAgent] Download error classified as: {error_type}")

                if error_type in UNRECOVERABLE:
                    print("[TranscriptAgent] Unrecoverable error — stopping immediately")
                    raise RuntimeError(USER_MESSAGES.get(error_type, last_error))

                if error_type in BACKOFF_RETRY and attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"[TranscriptAgent] Backing off {wait}s before retry attempt {attempt + 1}...")
                    time.sleep(wait)
                    continue

                if attempt == self.MAX_RETRIES:
                    print(f"[TranscriptAgent] All {self.MAX_RETRIES} download attempts failed")
                    raise RuntimeError(
                        USER_MESSAGES.get(error_type, last_error)
                    )

                # format_error / unknown on a non-final attempt: small fixed pause
                time.sleep(1)

        raise RuntimeError(f"Download failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _download_audio(self, url: str, output_dir: str) -> str:
        import shutil
        output_template = os.path.join(output_dir, "audio.%(ext)s")

        node_path   = shutil.which("node") or "/usr/bin/node"
        js_runtimes = {"node": {"path": node_path}} if os.path.exists(node_path) else {}

        print("[TranscriptAgent] Using player clients: android, ios, tv_embedded, web")
        if os.path.exists(node_path):
            print(f"[TranscriptAgent] JS runtime: {node_path}")
        else:
            print("[TranscriptAgent] JS runtime: not found, proceeding without")

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

        cookies_path = self._get_cookies_path()
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
            print(f"[TranscriptAgent] Cookies: loaded from {cookies_path}")
        else:
            print("[TranscriptAgent] Cookies: not set")

        print("[TranscriptAgent] Downloading audio stream...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise RuntimeError(str(e))

        audio_path = self._find_audio_file(output_dir)
        filename   = os.path.basename(audio_path)
        size_mb    = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"[TranscriptAgent] Audio file found: {filename}")
        print(f"[TranscriptAgent] File size: {size_mb:.1f} MB")

        if size_mb > self.MAX_FILE_SIZE_MB:
            raise RuntimeError(
                f"Audio file is {size_mb:.1f} MB — exceeds Groq Whisper's "
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
        print("[TranscriptAgent] Checking YOUTUBE_COOKIES environment variable...")
        content = os.environ.get("YOUTUBE_COOKIES", "").strip()
        if not content:
            print("[TranscriptAgent] YOUTUBE_COOKIES not set — proceeding without cookies")
            return None
        if os.path.exists(content):
            print("[TranscriptAgent] YOUTUBE_COOKIES is a file path — using directly")
            return content
        # Treat any non-path value as raw Netscape cookie content.
        print("[TranscriptAgent] YOUTUBE_COOKIES is cookie content — writing to temp file")
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
        last_error = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"[TranscriptAgent] Sending audio to Groq Whisper (attempt {attempt} of {self.MAX_RETRIES})...")
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
                print(f"[TranscriptAgent] ✅ Transcription complete — {len(transcript)} characters")
                return transcript

            except Exception as e:
                last_error = str(e)
                error_type = _classify(last_error)
                print(f"[TranscriptAgent] Transcription error classified as: {error_type}")

                if error_type in BACKOFF_RETRY and attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"[TranscriptAgent] Rate limited — backing off {wait}s...")
                    time.sleep(wait)
                    continue

                if attempt == self.MAX_RETRIES:
                    print(f"[TranscriptAgent] All transcription attempts failed")
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
