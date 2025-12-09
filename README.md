# Data Whisperer

Data Whisperer is a small Streamlit app that lets you upload a CSV and ask
plain-language questions about it.

Under the hood it:
- Inspects the columns and data types
- Runs basic summaries (row/column counts, numeric stats)
- Checks missing values by column
- Builds quick charts (distributions, top categories, simple trends)
- Uses a pluggable LLM client (stubbed for now) so you can later swap in a real model

The goal is to feel like you are talking to an analyst, not a spreadsheet.

---

## How to run it locally

```bash
cd ~/data-whisperer
source .venv/bin/activate
streamlit run frontend/app.py

```bash
