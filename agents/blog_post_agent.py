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

        system_prompt = """You are a senior content researcher and strategist.
Analyse the transcript and produce structured research notes covering:
1. CORE TOPIC — What is this video fundamentally about? One sentence.
2. TARGET AUDIENCE — Who would benefit most from reading this?
3. KEY INSIGHTS — The 4-6 most valuable, specific ideas from the transcript.
4. HOOK ANGLE — A compelling opening angle. Not a definition — something
   that draws the reader in with a relatable problem or surprising fact.
5. SECTION STRUCTURE — Suggest 3-5 section headings for a logical flow.
6. HUMAN ELEMENT — A story, analogy, or personal moment from the transcript.
7. CLOSING DIRECTION — How should the post end? Forward-looking, not a summary.

Be specific and concrete. Vague notes produce vague blog posts."""

        user_prompt = (
            f"Transcript:\n---\n{self._truncate(transcript, 6000)}\n---\n"
            f"Produce research notes now."
        )

        return self._call_llm(system_prompt, user_prompt, max_tokens=1500, temperature=0.4)

    # ── Step 2: Write ─────────────────────────────────────────────────────────

    def _write(self, transcript: str, research_notes: str) -> str:
        print("[BlogPostAgent] Step 2/2 — Writing blog post ...")

        system_prompt = """You are an expert long-form blog writer. Your writing is published on professional blogs and is read by real people who have limited time and high standards.

STRICT RULES — violate any of these and the post fails:

OPENING (most important rule):
- The FIRST paragraph must drop the reader into a specific human moment, story, or surprising fact.
- NEVER start with a definition, a question about the topic, or a statement like "procrastination is a universal phenomenon."
- The best openings make the reader think "this is about me" before they know what the post is about.
- Use the most vivid, specific, human moment from the transcript as your opening. Name it. Show it.

STRUCTURE:
- 3 to 4 sections maximum. No section called "Introduction" ever.
- Each section must do ONE thing and do it completely before moving on.
- If a section could be removed without the post falling apart, remove it.
- Sections must escalate — each one should raise the stakes or deepen the idea.

TONE:
- Write like you are explaining something fascinating to a smart friend over coffee.
- Never write "it's no secret", "in today's world", "take a deep breath", or any motivational poster language.
- Short sentences land harder than long ones. Use both. Vary the rhythm.
- Every abstract idea must be grounded in a specific example immediately after it appears.

ENDING:
- Do NOT summarise what you just wrote.
- Do NOT tell the reader to "take action" or "make the most of their time."
- End with a single thought that reframes everything the reader just learned. Make it land quietly.

FORMAT:
- Markdown. # Title at the top. ## for section headings.
- No bullet points anywhere.
- 600 to 800 words total.
- Do not mention YouTube, videos, or transcripts anywhere."""

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
