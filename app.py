import io
import os
import builtins
import streamlit as st
from docx import Document
from orchestrator import Orchestrator


def blog_post_to_docx(markdown_text: str) -> bytes:
    """Convert a markdown blog post string into a .docx file in memory."""
    try:
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
    except Exception as e:
        print(f"[blog_post_to_docx] Failed to generate Word document: {e}")
        return b""

st.set_page_config(
    page_title="YouTube to Blog Post",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("📝 YT Blog Agent")
    st.caption("YouTube to Blog Post pipeline")
    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
    - ✅ **Agent 1** — Transcript
    - ✅ **Agent 2** — Blog Post
    """)
    st.divider()
    api_status = "✅ Set" if os.environ.get("GROQ_API_KEY") else "❌ Missing"
    st.caption(f"Groq API Key: {api_status}")
    st.caption("Groq Whisper + Llama 3.3 70B")
    st.divider()
    st.markdown("### Transcript Source")
    st.caption("⚡ Captions API — instant, no download")
    st.caption("🎙️ Audio download — fallback for videos without captions")

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

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it in Railway Variables.")
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

            # Capture all agent print statements into the live log panel
            original_print = builtins.print

            def captured_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                if msg.startswith("[Transcript") or msg.startswith("[BlogPost") or msg.startswith("[Orchestrator"):
                    append_log(msg)
                original_print(*args, **kwargs)

            builtins.print = captured_print
            try:
                result = orch.run(url.strip())
            finally:
                builtins.print = original_print

            status.update(label="Done!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    # ── Output ────────────────────────────────────────────────────────────────
    blog_post         = result["blog_post"]
    transcript        = result["transcript"]
    transcript_source = result.get("transcript_source", "audio_download")
    docx_bytes        = blog_post_to_docx(blog_post)

    st.divider()

    tab1, tab2 = st.tabs(["Blog Post", "Raw Transcript"])

    with tab1:
        st.markdown(blog_post)
        st.download_button(
            label="Download as Word Document",
            data=docx_bytes,
            file_name="blog_post.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    with tab2:
        source_badge = (
            "⚡ Via captions" if transcript_source == "captions_api"
            else "🎙️ Via audio transcription"
        )
        st.caption(source_badge)
        st.text_area(
            label="Transcript",
            value=transcript,
            height=400,
            disabled=True,
        )
        st.caption(f"{len(transcript.split())} words · {len(transcript)} characters")
