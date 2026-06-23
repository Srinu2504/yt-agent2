"""
Orchestrator
------------
Coordinates TranscriptAgent and BlogPostAgent in sequence.
Receives a YouTube URL, returns a finished blog post.

Flow:
    YouTube URL → TranscriptAgent → transcript → BlogPostAgent → blog post
"""

from agents.transcript_agent import TranscriptAgent
from agents.blog_post_agent import BlogPostAgent


class Orchestrator:
    """
    Thin coordinator — holds no business logic of its own.
    Just wires the two agents together in the right order.

    Usage:
        orchestrator = Orchestrator()
        blog_post = orchestrator.run("https://www.youtube.com/watch?v=...")
    """

    def __init__(self):
        print("[Orchestrator] Initialising agents …")
        self.transcript_agent = TranscriptAgent()
        self.blog_post_agent = BlogPostAgent()
        print("[Orchestrator] Ready ✓")

    def run(self, youtube_url: str) -> dict:
        """
        Main entry point called by the Streamlit UI.

        Args:
            youtube_url: Any valid YouTube video URL.

        Returns:
            A dict with keys:
                - 'transcript': the raw transcript string
                - 'blog_post':  the finished blog post in markdown
        """
        print(f"\n[Orchestrator] Starting pipeline for: {youtube_url}")

        # Step 1 — Transcribe
        print("[Orchestrator] Calling TranscriptAgent …")
        transcript = self.transcript_agent.run(youtube_url)
        print(f"[Orchestrator] Transcript received ({len(transcript)} chars)")

        # Step 2 — Write blog post
        print("[Orchestrator] Calling BlogPostAgent …")
        blog_post = self.blog_post_agent.run(transcript)
        print(f"[Orchestrator] Blog post received ({len(blog_post)} chars)")

        print("[Orchestrator] Pipeline complete ✓")

        return {
            "transcript": transcript,
            "blog_post": blog_post,
        }


# ----------------------------------------------------------------------
# Standalone test — tests the full pipeline end to end
#   python orchestrator.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    TEST_URL = "https://www.youtube.com/watch?v=arj7oStGLkU"

    print("=" * 60)
    print("Orchestrator — full pipeline test")
    print("=" * 60)

    try:
        orchestrator = Orchestrator()
        result = orchestrator.run(TEST_URL)

        print("\n--- TRANSCRIPT PREVIEW (first 300 chars) ---")
        print(result["transcript"][:300])

        print("\n--- BLOG POST PREVIEW (first 500 chars) ---")
        print(result["blog_post"][:500])

        print("\n✅ Orchestrator test PASSED — full pipeline working")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Orchestrator test FAILED: {e}")
        sys.exit(1)