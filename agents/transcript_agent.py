"""
TranscriptAgent
---------------
Responsibility: Given a YouTube URL, download the audio and return
a plain-text transcript using Groq's Whisper API.

Dependencies: yt-dlp, groq, python-dotenv
"""

import os
import tempfile
import yt_dlp
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class TranscriptAgent:
    """
    Agent 1 — Downloads YouTube audio and transcribes it via Groq Whisper.

    Usage:
        agent = TranscriptAgent()
        transcript = agent.run("https://www.youtube.com/watch?v=...")
    """

    # Groq Whisper supports files up to 25 MB.
    # yt-dlp will pick the smallest audio stream to stay under this.
    MAX_FILE_SIZE_MB = 24

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        print("[TranscriptAgent] Initialised ✓")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, youtube_url: str) -> str:
        """
        Main entry point called by the Orchestrator.

        Args:
            youtube_url: Any valid YouTube video URL.

        Returns:
            Plain-text transcript string.

        Raises:
            ValueError: If the URL is empty or obviously invalid.
            RuntimeError: If download or transcription fails.
        """
        self._validate_url(youtube_url)

        print(f"[TranscriptAgent] Processing: {youtube_url}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = self._download_audio(youtube_url, tmp_dir)
            transcript = self._transcribe(audio_path)

        print(f"[TranscriptAgent] Done — {len(transcript)} characters transcribed.")
        return transcript

    # ------------------------------------------------------------------
    # Step 1 — Download
    # ------------------------------------------------------------------

    def _download_audio(self, url: str, output_dir: str) -> str:
        """
        Downloads the audio track of a YouTube video using yt-dlp.

        Picks the smallest available audio format to stay within
        Groq Whisper's 25 MB file-size limit. Saves as .mp3.

        Returns:
            Absolute path to the downloaded audio file.
        """
        output_template = os.path.join(output_dir, "audio.%(ext)s")

        ydl_opts = {
            # Best audio quality but capped at a manageable size.
            # 'worstaudio' keeps files small; for longer videos this
            # matters since Groq Whisper has a 25 MB hard limit.
            "format": "worstaudio/bestaudio",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",   # 64 kbps keeps files small
                }
            ],
            # Suppress yt-dlp's own console output so our logs stay clean
            "quiet": True,
            "no_warnings": True,
        }

        print("[TranscriptAgent] Step 1/2 — Downloading audio …")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as e:
            raise RuntimeError(f"Audio download failed: {e}") from e

        # Find the file yt-dlp just wrote (extension may vary)
        audio_path = self._find_audio_file(output_dir)

        # Sanity-check file size
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"[TranscriptAgent] Audio downloaded → {size_mb:.1f} MB")

        if size_mb > self.MAX_FILE_SIZE_MB:
            raise RuntimeError(
                f"Audio file is {size_mb:.1f} MB, which exceeds Groq Whisper's "
                f"25 MB limit. Try a shorter video."
            )

        return audio_path

    def _find_audio_file(self, directory: str) -> str:
        """Finds the audio file yt-dlp wrote in the given directory."""
        audio_extensions = {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".webm"}
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in audio_extensions:
                return os.path.join(directory, filename)
        raise RuntimeError(
            "Audio download appeared to succeed but no audio file was found. "
            "Check that ffmpeg is installed."
        )

    # ------------------------------------------------------------------
    # Step 2 — Transcribe
    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: str) -> str:
        """
        Sends the audio file to Groq's Whisper API and returns the transcript.

        Uses whisper-large-v3-turbo — Groq's fastest Whisper model,
        well-suited for English YouTube content.

        Returns:
            Plain-text transcript string.
        """
        print("[TranscriptAgent] Step 2/2 — Transcribing via Groq Whisper …")
        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="text",   # plain string, not JSON
                    language="en",
                )
        except Exception as e:
            raise RuntimeError(f"Groq Whisper transcription failed: {e}") from e

        # Groq returns the transcript directly as a string when
        # response_format="text". Clean up any leading/trailing whitespace.
        transcript = str(response).strip()

        if not transcript:
            raise RuntimeError(
                "Whisper returned an empty transcript. "
                "The video may have no speech or be in an unsupported language."
            )

        return transcript

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_url(self, url: str) -> None:
        """Basic URL sanity check before hitting the network."""
        if not url or not isinstance(url, str):
            raise ValueError("YouTube URL must be a non-empty string.")
        url = url.strip()
        if "youtube.com" not in url and "youtu.be" not in url:
            raise ValueError(
                f"'{url}' does not look like a YouTube URL. "
                "Expected youtube.com or youtu.be."
            )


# ----------------------------------------------------------------------
# Standalone test — run this file directly to verify the agent works
# before wiring it into the Orchestrator.
#
#   python agents/transcript_agent.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Short test video (60-second TED-Ed clip — safe for testing)
    TEST_URL = "https://www.youtube.com/watch?v=arj7oStGLkU"

    print("=" * 60)
    print("TranscriptAgent — standalone test")
    print("=" * 60)

    try:
        agent = TranscriptAgent()
        transcript = agent.run(TEST_URL)

        print("\n--- TRANSCRIPT PREVIEW (first 500 chars) ---")
        print(transcript[:500])
        print("..." if len(transcript) > 500 else "")
        print(f"\nTotal length: {len(transcript)} characters")
        print("\n✅ TranscriptAgent test PASSED")
        sys.exit(0)

    except EnvironmentError as e:
        print(f"\n❌ Environment error: {e}")
        print("Fix: Add GROQ_API_KEY=your_key_here to a .env file")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        sys.exit(1)
