import os
import sys
from typing import List, Tuple

import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.engines.insight_engine import InsightEngine  # noqa: E402

st.set_page_config(
    page_title="Data Whisperer",
    page_icon="📊",
    layout="wide",
)

engine = InsightEngine()


def get_chat_history() -> List[Tuple[str, str]]:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history


def get_dataframe() -> pd.DataFrame:
    return st.session_state.get("df", pd.DataFrame())


def main() -> None:
    st.title("📊 Data Whisperer")

    st.write(
        "Upload a CSV, then ask plain-language questions about your data. "
        "Right now the LLM layer is a stub, but the analysis pipeline is wired."
    )

    st.sidebar.header("Upload data")
    uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            df = pd.DataFrame()
        st.session_state.df = df
    else:
        df = get_dataframe()

    if df.empty:
        st.info("Upload a CSV in the sidebar to get started.")
        return

    with st.expander("Preview data", expanded=False):
        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(20))

    history = get_chat_history()

    st.subheader("Ask a question about this dataset")

    user_input = st.chat_input("Ask a question (or leave blank for an overview)...")

    if user_input is not None:
        q = user_input.strip()
        history.append(("user", q or "[overview request]"))

        with st.spinner("Thinking..."):
            result = engine.analyze(df, q, history=history)

        history.append(("assistant", result.answer))
        st.session_state.chat_history = history
        st.session_state.last_chart = result.chart

    for role, text in history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)

    if "last_chart" in st.session_state and st.session_state.last_chart is not None:
        st.altair_chart(st.session_state.last_chart, use_container_width=True)


if __name__ == "__main__":
    main()

