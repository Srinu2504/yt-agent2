import io
import os
import streamlit as st
from docx import Document
from docx.shared import Pt
from orchestrator import Orchestrator


def blog_post_to_docx(markdown_text: str) -> bytes:
    """Convert a markdown blog post string into a .docx file in memory."""
    doc = Document()
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped == "":
            doc.add_paragraph("")
        else:
            doc.add_paragraph(stripped)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

st.set_page_config(
    page_title="YouTube to Blog Post",
    page_icon="📝",
    layout="centered",
)

st.title("📝 YouTube to Blog Post")
st.caption("Paste a YouTube URL and get a publish-ready blog post.")

# ── Input ─────────────────────────────────────────────────────────────────────
url = st.text_input(
    label="YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Any public YouTube video with speech will work.",
)

generate = st.button("Generate Blog Post", type="primary", use_container_width=True)

# ── API key check ─────────────────────────────────────────────────────────────
if not os.environ.get("GROQ_API_KEY"):
    st.warning("GROQ_API_KEY is not set. Add it in Railway → Variables.")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if generate:
    if not url.strip():
        st.warning("Please paste a YouTube URL first.")
        st.stop()

    log_lines = []

    def append_log(msg: str):
        log_lines.append(msg)
        log_area.code("\n".join(log_lines), language=None)

    try:
        with st.status("Running pipeline...", expanded=True) as status:
            log_area = st.empty()

            append_log("Initialising agents...")
            orch = Orchestrator()

            # Patch print statements by overriding run methods
            original_t_run = orch.transcript_agent.run
            original_b_run = orch.blog_post_agent.run

            def t_run_patched(youtube_url):
                append_log("Downloading and transcribing audio...")
                result = original_t_run(youtube_url)
                append_log(f"Transcript ready — {len(result):,} characters")
                return result

            def b_run_patched(transcript):
                append_log("Researching content...")
                append_log("Writing blog post...")
                result = original_b_run(transcript)
                append_log(f"Blog post ready — {len(result):,} characters")
                return result

            orch.transcript_agent.run = t_run_patched
            orch.blog_post_agent.run  = b_run_patched

            result = orch.run(url.strip())
            status.update(label="Done!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    # ── Output ────────────────────────────────────────────────────────────────
    blog_post  = result["blog_post"]
    transcript = result["transcript"]

    st.divider()

    tab1, tab2 = st.tabs(["Blog Post", "Raw Transcript"])

    with tab1:
        st.markdown(blog_post)
        st.divider()
        st.download_button(
            label="Download as .md",
            data=blog_post,
            file_name="blog_post.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            label="Download as Word Document",
            data=blog_post_to_docx(blog_post),
            file_name="blog_post.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    with tab2:
        st.text_area(
            label="Transcript",
            value=transcript,
            height=400,
            disabled=True,
        )
        st.caption(f"{len(transcript.split())} words · {len(transcript)} characters")
