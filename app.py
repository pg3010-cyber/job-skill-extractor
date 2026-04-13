"""Streamlit app for the Job Description Skill Extractor."""

import streamlit as st

import extractor

SAMPLE_TEXT = (
    "We need a Python developer with experience in Machine Learning, SQL, "
    "and Docker. Knowledge of AWS is a plus."
)

st.set_page_config(page_title="Job Skill Extractor", page_icon="🔍")

st.title("Job Skill Extractor")
st.caption("Paste a job description and extract technical skills with spaCy NLP.")

if "job_text" not in st.session_state:
    st.session_state.job_text = ""


def use_sample_text():
    """Load a quick demo example into the input area."""
    st.session_state.job_text = SAMPLE_TEXT


st.text_area(
    "Job Description",
    key="job_text",
    height=200,
    placeholder="Paste a job description here...",
)

left_button, right_button = st.columns(2)
with left_button:
    st.button("Try Sample Input", on_click=use_sample_text, use_container_width=True)
with right_button:
    run_extraction = st.button("Extract Skills", type="primary", use_container_width=True)

if run_extraction:
    input_text = st.session_state.job_text.strip()

    if not input_text:
        st.warning("Please enter a job description first.")
    else:
        results = extractor.extract_skills(input_text)
        skills = results["skills"]
        frequency = results["frequency"]

        st.metric("Total Skills Found", len(skills))

        skill_col, chart_col = st.columns(2)
        with skill_col:
            st.subheader("Extracted Skills")
            if skills:
                for skill in skills:
                    st.markdown(f"- {skill}")
            else:
                st.info("No skills were detected in this text.")

        with chart_col:
            st.subheader("Skill Frequency")
            if frequency:
                chart_rows = [
                    {"Skill": skill, "Frequency": count}
                    for skill, count in frequency.items()
                ]
                st.bar_chart(chart_rows, x="Skill", y="Frequency")
            else:
                st.info("No frequency data to plot yet.")

        st.subheader("POS Tags")
        pos_rows = [{"Word": word, "POS": pos} for word, pos in results["pos_tags"]]
        st.dataframe(pos_rows, use_container_width=True)

        st.subheader("Dependency Parse Output")
        dep_rows = [
            {"Token": token, "Dependency": dep_label, "Head": head}
            for token, dep_label, head in results["dep_info"]
        ]
        st.dataframe(dep_rows, use_container_width=True)

st.markdown("---")
st.caption("Built with spaCy + Streamlit")
