"""
BlogPostAgent
-------------
Responsibility: Given a plain-text transcript, run an internal
research pass and then write a polished blog post — both steps
using Groq's Llama 3.3 70B model.

This is ONE agent with TWO internal LLM calls.
There is no separate ResearchAgent.

Dependencies: groq, python-dotenv
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class BlogPostAgent:
    """
    Agent 2 — Researches a transcript internally, then writes a blog post.

    Internal flow:
        1. _research(transcript)  → research_notes  (LLM call 1)
        2. _write(transcript, research_notes) → blog_post (LLM call 2)

    Usage:
        agent = BlogPostAgent()
        blog_post = agent.run(transcript)
    """

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        print("[BlogPostAgent] Initialised ✓")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, transcript: str) -> str:
        """
        Main entry point called by the Orchestrator.

        Args:
            transcript: Plain-text transcript from TranscriptAgent.

        Returns:
            A fully written blog post as a markdown string.

        Raises:
            ValueError: If the transcript is empty.
            RuntimeError: If either LLM call fails.
        """
        self._validate_transcript(transcript)

        print(f"[BlogPostAgent] Received transcript ({len(transcript)} chars)")

        # Step 1 — Internal research pass
        research_notes = self._research(transcript)

        # Step 2 — Write the blog post
        blog_post = self._write(transcript, research_notes)

        print(f"[BlogPostAgent] Done — blog post written ({len(blog_post)} chars)")
        return blog_post

    # ------------------------------------------------------------------
    # Step 1 — Internal research
    # ------------------------------------------------------------------

    def _research(self, transcript: str) -> str:
        """
        First LLM call: analyse the transcript and produce structured
        research notes that will guide the blog post.

        This is an internal step — the notes are never shown to the user.
        """
        print("[BlogPostAgent] Step 1/2 — Running internal research …")

        system_prompt = """You are a senior content researcher and strategist.
Your job is to deeply analyse a YouTube video transcript and produce
structured research notes that a writer will use to craft a blog post.

Your research notes must include:

1. CORE TOPIC — What is this video fundamentally about? One sentence.

2. TARGET AUDIENCE — Who would benefit most from reading a blog post
   about this? Be specific (e.g. "early-stage founders", "patients
   recently diagnosed with X", "developers learning Python").

3. KEY INSIGHTS — The 4-6 most valuable, specific ideas from the transcript.
   Not summaries — actual insights a reader would find genuinely useful.
   Quote or closely paraphrase the most compelling moments.

4. HOOK ANGLE — A compelling opening angle for the blog post. Not a
   definition or background. Something that draws the reader in with
   a relatable problem, surprising fact, or vivid scenario.

5. SECTION STRUCTURE — Suggest 3-5 section headings that would make
   a logical, engaging flow for the blog post. Each heading should
   earn the next — no filler sections.

6. HUMAN ELEMENT — Is there a story, analogy, or personal moment in
   the transcript that can make the post feel real and relatable?
   Identify it specifically.

7. CLOSING DIRECTION — How should the post end? Not a summary.
   A forward-looking thought, a call to action, or a reason to care
   going forward.

Be specific and concrete. Vague notes produce vague blog posts."""

        user_prompt = f"""Here is the video transcript to analyse:

---
{transcript}
---

Produce your structured research notes now."""

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,   # Lower = more focused analysis
                max_tokens=1500,
            )
        except Exception as e:
            raise RuntimeError(f"Research LLM call failed: {e}") from e

        research_notes = response.choices[0].message.content.strip()

        print(f"[BlogPostAgent] Research notes ready ({len(research_notes)} chars)")
        return research_notes

    # ------------------------------------------------------------------
    # Step 2 — Write the blog post
    # ------------------------------------------------------------------

    def _write(self, transcript: str, research_notes: str) -> str:
        """
        Second LLM call: use the transcript + research notes to write
        a polished, publish-ready blog post.

        The writing style is modelled on the three reference blog posts
        the mentor shared — accessible, concrete, human, purposeful.
        """
        print("[BlogPostAgent] Step 2/2 — Writing blog post …")

        system_prompt = """You are an expert blog writer who produces
publish-ready long-form content. You write in the style of the best
professional blogs — accessible, specific, human, and purposeful.

WRITING PRINCIPLES (follow all of these):

OPENING HOOK
- Never start with a definition or background.
- Open with a relatable problem, a vivid scenario, or a surprising fact
  that makes the reader feel seen before you explain anything.
- The first paragraph should make someone think "this is about me."

STRUCTURE
- Use clear H2 section headings (## in markdown).
- Each section should earn the next — no filler, no repetition.
- 3-5 sections is the right length. Do not pad.

TONE & LANGUAGE
- Write for a general reader, not a specialist.
- Every technical term must be explained in plain words immediately
  after it appears.
- Use short sentences alongside longer ones. Vary the rhythm.
- Be concrete and specific — real examples, not vague generalisations.
- Never write "In conclusion" or "In summary." Just end well.

HUMAN ELEMENT
- Include at least one analogy, story, or vivid real-world example
  that makes an abstract idea feel tangible.
- Write as if you are explaining something fascinating to a smart
  friend, not presenting a report.

ENDING
- Do not summarise what you just said.
- End with a forward-looking thought, a reason to care, or a quiet
  call to action that feels earned.

FORMAT
- Output in clean markdown.
- Start with a # Title
- Use ## for section headings
- No bullet-point dumps — write in flowing paragraphs.
- Target length: 600-900 words (not counting the title).

Do not mention that this is based on a YouTube video or transcript."""

        user_prompt = f"""Here are the research notes for this blog post:

---RESEARCH NOTES---
{research_notes}
---END NOTES---

Here is the original transcript for additional detail and quotes:

---TRANSCRIPT---
{transcript[:3000]}
---END TRANSCRIPT---

Now write the complete, publish-ready blog post in markdown."""

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,   # Higher = more creative writing
                max_tokens=2000,
            )
        except Exception as e:
            raise RuntimeError(f"Writing LLM call failed: {e}") from e

        blog_post = response.choices[0].message.content.strip()

        if not blog_post:
            raise RuntimeError("LLM returned an empty blog post.")

        return blog_post

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_transcript(self, transcript: str) -> None:
        if not transcript or not isinstance(transcript, str):
            raise ValueError("Transcript must be a non-empty string.")
        if len(transcript.strip()) < 100:
            raise ValueError(
                f"Transcript is too short ({len(transcript)} chars). "
                "Minimum 100 characters required."
            )


# ----------------------------------------------------------------------
# Standalone test — run this file directly to verify the agent works
# before wiring it into the Orchestrator.
#
#   python agents\blog_post_agent.py
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Hardcoded transcript sample — the same TED-Ed video from
    # TranscriptAgent's test, pasted in so we don't need to re-download.
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
    They think: I have plenty of time, I'll get to this later.

    The frustrating thing about procrastination isn't that it makes you
    lazy. It's that it makes you feel terrible. There's the Panic Monster —
    that's what wakes you up at 2am when the deadline is tomorrow. Without
    the Panic Monster, nothing would ever get done.

    But here's the problem: the Panic Monster only shows up when there's an
    immediate deadline. What about the procrastination that doesn't have a
    deadline? The big goals — starting a business, writing a book, getting
    healthy, fixing relationships — these have no deadline. And so the Panic
    Monster never appears. And nothing ever happens.

    We all have a finite amount of time. And that's what should scare us more
    than any deadline. The question isn't whether you procrastinate. It's
    whether you're aware of when you're doing it — and whether you're letting
    it steal the things that matter most to you.
    """

    print("=" * 60)
    print("BlogPostAgent — standalone test")
    print("=" * 60)

    try:
        agent = BlogPostAgent()
        blog_post = agent.run(TEST_TRANSCRIPT)

        print("\n--- BLOG POST OUTPUT ---\n")
        print(blog_post)
        print(f"\nTotal length: {len(blog_post)} characters")
        print("\n✅ BlogPostAgent test PASSED")
        sys.exit(0)

    except EnvironmentError as e:
        print(f"\n❌ Environment error: {e}")
        print("Fix: Add GROQ_API_KEY=your_key_here to your .env file")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        sys.exit(1)
