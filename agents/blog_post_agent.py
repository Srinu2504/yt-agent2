"""
BlogPostAgent
-------------
Responsibility: Given a plain-text transcript, run an internal
research pass and then write a polished blog post.

Agentic features:
- Research failure is non-fatal (falls back to raw transcript)
- Retry with backoff on rate limits
- Transcript truncated intelligently for long videos
- Both LLM calls independently retried
"""
import os
import time
from groq import Groq, AuthenticationError
from dotenv import load_dotenv

load_dotenv()


class BlogPostAgent:
    """
    Agent 2 — Researches a transcript internally then writes a blog post.

    Internal flow:
        1. _research(transcript) -> research_notes  (LLM call 1, temp=0.4)
        2. _write(transcript, research_notes) -> blog_post (LLM call 2, temp=0.7)

    If research fails, _write() uses the raw transcript only.
    """

    MODEL       = "llama-3.3-70b-versatile"
    MAX_RETRIES = 3

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found.")
        self.client = Groq(api_key=api_key)
        print("[BlogPostAgent] Initialised")

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, transcript: str) -> str:
        self._validate_transcript(transcript)
        print(f"[BlogPostAgent] Received transcript ({len(transcript)} chars)")

        # Step 1 — Research (non-fatal if it fails)
        try:
            research_notes = self._research(transcript)
        except Exception as e:
            print(f"[BlogPostAgent] Research failed ({e}) — writing from transcript only")
            research_notes = ""

        # Step 2 — Write
        blog_post = self._write(transcript, research_notes)
        print(f"[BlogPostAgent] Done — {len(blog_post)} chars")
        return blog_post

    # ── Step 1: Research ──────────────────────────────────────────────────────

    def _research(self, transcript: str) -> str:
        print("[BlogPostAgent] Step 1/2 — Running internal research ...")

        system_prompt = """You are a content researcher preparing notes for a blog writer. Your job is to extract the most compelling raw material from this transcript so the writer never has to guess what to use.

Produce research notes with these 6 sections:

BEST OPENING MOMENT:
Find the single most specific, vivid, human moment in the transcript. It must be a concrete scene or story — not a concept or idea. Describe it in 2-3 sentences exactly as it happened. This will be the opening of the blog post.

CORE IDEA:
One sentence. What is this transcript fundamentally arguing or revealing?

KEY CONCEPTS (3 maximum):
The 3 most important ideas. For each one: name it, explain it in one sentence, give the specific example or analogy from the transcript that illustrates it.

MOST MEMORABLE LINE:
The single most quotable or striking sentence from the transcript. Copy it exactly.

HUMAN STORIES:
List every specific personal story, anecdote, or real example mentioned. Include names, numbers, and details. These are the writer's raw material — do not summarise them, describe them precisely.

ENDING DIRECTION:
What thought or idea from the transcript would leave a reader thinking after they close the tab? Not a call to action. A reframing or quiet observation.

Be specific. Vague research notes produce vague blog posts. If the transcript mentions a number, include it. If it mentions a name, include it. If it describes a scene, describe it the same way."""

        user_prompt = (
            f"Transcript:\n---\n{self._truncate(transcript, 6000)}\n---\n"
            f"Produce research notes now."
        )

        return self._call_llm(system_prompt, user_prompt, max_tokens=1500, temperature=0.4)

    # ── Step 2: Write ─────────────────────────────────────────────────────────

    def _write(self, transcript: str, research_notes: str) -> str:
        print("[BlogPostAgent] Step 2/2 — Writing blog post ...")

        system_prompt = """You are a professional blog writer. You have been given research notes and a transcript excerpt. Your job is to write a blog post that uses the specific material in those notes — the stories, the numbers, the examples. Do not invent examples. Do not use generic scenarios. Every claim must come from the research notes or transcript.

OPENING — non-negotiable:
Use the BEST OPENING MOMENT from the research notes as your opening paragraph. Make it vivid and specific. Write in second person. Do not start with a question. Do not start with "Imagine". Drop straight into the moment.

STRUCTURE:
Exactly 3 sections with ## headings. Each heading names the specific idea in that section — never "Introduction", "Strategies", or "Conclusion". Sections escalate — each one deepens the idea from the previous one.

WRITING RULES:
Second person throughout — "you", never "I" or "we". Short sentences hit harder than long ones — use both. Every abstract idea gets one specific example from the research notes immediately after it. No bullet points. Flowing paragraphs only.

BANNED PHRASES:
"it is no secret", "in today's world", "take a deep breath", "make the most of", "in conclusion", "to summarize", "dive deep", "imagine having", "at the end of the day", "it is worth noting".

ENDING:
One short paragraph. No questions. No calls to action. Use the ENDING DIRECTION from the research notes. Make it land quietly — a thought the reader carries with them.

FORMAT:
# Title. ## for 3 section headings. 650 to 800 words. No mention of YouTube, videos, or transcripts."""

        if research_notes:
            user_prompt = (
                f"Research notes:\n{self._truncate(research_notes, 3000)}\n\n"
                f"Transcript excerpt:\n{self._truncate(transcript, 3000)}\n\n"
                f"Write the complete publish-ready blog post now."
            )
        else:
            user_prompt = (
                f"Transcript:\n{self._truncate(transcript, 4000)}\n\n"
                f"Write a complete publish-ready blog post from this transcript now."
            )

        return self._call_llm(system_prompt, user_prompt, max_tokens=2000, temperature=0.7)

    # ── LLM call with retry ───────────────────────────────────────────────────

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        last_error = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120,
                )
                result = response.choices[0].message.content.strip()
                if not result:
                    raise RuntimeError("LLM returned empty response.")
                return result

            except AuthenticationError:
                raise RuntimeError("Invalid Groq API key.")

            except Exception as e:
                last_error = str(e)
                print(f"[BlogPostAgent] LLM attempt {attempt} failed: {last_error[:60]}")

                if ("429" in last_error or "rate limit" in last_error.lower()) \
                        and attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"[BlogPostAgent] Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue

                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(f"LLM failed after {self.MAX_RETRIES} attempts: {last_error}")

        raise RuntimeError(f"LLM call failed: {last_error}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        front = int(max_chars * 0.8)
        back  = max_chars - front
        return text[:front] + "\n\n[...middle omitted...]\n\n" + text[-back:]

    def _validate_transcript(self, transcript: str) -> None:
        if not transcript or not isinstance(transcript, str):
            raise ValueError("Transcript must be a non-empty string.")
        if len(transcript.strip()) < 100:
            raise ValueError("Transcript too short (minimum 100 characters).")


if __name__ == "__main__":
    import sys
    TEST_TRANSCRIPT = """
    So in college, I was a government major, which means I had to write a lot
    of papers. Now, when a normal student writes a paper, they might spread
    their work out a little like this. So, you know, you get started maybe a
    little slowly, but you get enough done in the first week that with some
    heavier days later on, everything gets done and things stay civil.
    And I would want to do that like that. That would be the plan. I would
    have it all ready to go, but then actually the paper would come along,
    and then I would kind of do this. And that would happen every single time.
    So I have a hypothesis that there are two kinds of procrastinators.
    The first type knows what they are. They look at the paper and they say:
    I should be working on this right now, but I'm going to watch YouTube
    instead. The second type doesn't realise how bad they are at it yet.
    The frustrating thing about procrastination isn't that it makes you
    lazy. It's that it makes you feel terrible. There's the Panic Monster —
    that's what wakes you up at 2am when the deadline is tomorrow. Without
    the Panic Monster, nothing would ever get done.
    We all have a finite amount of time. The question isn't whether you
    procrastinate. It's whether you're aware of when you're doing it.
    """
    print("=" * 60)
    print("BlogPostAgent — standalone test")
    print("=" * 60)
    try:
        agent     = BlogPostAgent()
        blog_post = agent.run(TEST_TRANSCRIPT)
        print("\n--- BLOG POST OUTPUT ---\n")
        print(blog_post[:1000])
        print(f"\nTotal: {len(blog_post)} characters")
        print("\nBlogPostAgent test PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        sys.exit(1)
