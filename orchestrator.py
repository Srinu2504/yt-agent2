"""
Orchestrator
------------
Thin coordinator — no business logic.
Wires TranscriptAgent and BlogPostAgent in sequence.

Flow:
    YouTube URL -> TranscriptAgent -> transcript -> BlogPostAgent -> blog post
"""
from agents.transcript_agent import TranscriptAgent
from agents.blog_post_agent  import BlogPostAgent


class Orchestrator:
    """
    Thin coordinator — holds no business logic of its own.

    Usage:
        orch   = Orchestrator()
        result = orch.run("https://www.youtube.com/watch?v=...")
        # result = {"transcript": "...", "blog_post": "..."}
    """

    def __init__(self):
        print("[Orchestrator] Initialising agents ...")
        self.transcript_agent = TranscriptAgent()
        self.blog_post_agent  = BlogPostAgent()
        print("[Orchestrator] Ready")

    def run(self, youtube_url: str) -> dict:
        """
        Returns:
            dict with keys:
                'transcript'        — raw transcript string
                'blog_post'         — finished blog post in Markdown
                'video_id'          — 11-character YouTube video ID
                'title'             — video title from yt-dlp metadata
                'transcript_source' — 'captions_api' or 'audio_download'
        """
        print(f"\n[Orchestrator] Starting pipeline for: {youtube_url}")

        print("[Orchestrator] Stage 1 — TranscriptAgent")
        agent_result      = self.transcript_agent.run(youtube_url)
        transcript        = agent_result["transcript"]
        video_id          = agent_result["video_id"]
        title             = agent_result["title"]
        transcript_source = getattr(self.transcript_agent, "last_source", "audio_download")
        print(f"[Orchestrator] Transcript ready ({len(transcript)} chars, source: {transcript_source})")

        print("[Orchestrator] Stage 2 — BlogPostAgent")
        blog_post = self.blog_post_agent.run(transcript)
        print(f"[Orchestrator] Blog post ready ({len(blog_post)} chars)")

        print("[Orchestrator] Pipeline complete")
        return {
            "transcript":        transcript,
            "blog_post":         blog_post,
            "video_id":          video_id,
            "title":             title,
            "transcript_source": transcript_source,
        }


if __name__ == "__main__":
    import sys
    TEST_URL = "https://www.youtube.com/watch?v=arj7oStGLkU"
    print("=" * 60)
    print("Orchestrator — full pipeline test")
    print("=" * 60)
    try:
        orch   = Orchestrator()
        result = orch.run(TEST_URL)
        print("\n--- TRANSCRIPT (first 300 chars) ---")
        print(result["transcript"][:300])
        print("\n--- BLOG POST (first 500 chars) ---")
        print(result["blog_post"][:500])
        print("\nOrchestrator test PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\nOrchestrator test FAILED: {e}")
        sys.exit(1)
