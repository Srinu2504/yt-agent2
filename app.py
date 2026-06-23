import streamlit as st
from orchestrator import Orchestrator

st.set_page_config(
    page_title="YouTube to Blog Post",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 YouTube to Blog Post")
st.caption("Paste a YouTube URL and get a publish-ready blog post in seconds.")

# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------

url = st.text_input(
    label="YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Any public YouTube video with speech will work."
)

generate = st.button("Generate Blog Post", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

if generate:
    if not url.strip():
        st.warning("Please paste a YouTube URL first.")
        st.stop()

    try:
        with st.status("Running pipeline...", expanded=True) as status:

            st.write("Initialising agents...")
            orchestrator = Orchestrator()

            st.write("Downloading and transcribing audio...")
            # Patch TranscriptAgent to report progress via st.write
            original_run = orchestrator.transcript_agent.run

            def run_with_progress(youtube_url):
                result = original_run(youtube_url)
                st.write(f"Transcript ready — {len(result):,} characters")
                return result

            orchestrator.transcript_agent.run = run_with_progress

            st.write("Researching and writing blog post...")
            result = orchestrator.run(url.strip())

            status.update(label="Done!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    blog_post = result["blog_post"]
    transcript = result["transcript"]

    st.divider()

    # Blog post tab and transcript tab
    tab1, tab2 = st.tabs(["Blog Post", "Raw Transcript"])

    with tab1:
        st.markdown(blog_post)
        st.divider()
        st.download_button(
            label="Download as .md",
            data=blog_post,
            file_name="blog_post.md",
            mime="text/markdown",
            use_container_width=True
        )

    with tab2:
        st.text_area(
            label="Transcript",
            value=transcript,
            height=400,
            disabled=True
        )