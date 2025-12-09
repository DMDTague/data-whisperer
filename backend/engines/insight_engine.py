from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd

from backend.llm.deepseek_client import DeepSeekClient


@dataclass
class InsightResult:
    """Structured result from the engine."""
    answer: str
    chart: Optional[alt.Chart] = None
    debug: Optional[Dict[str, Any]] = None


class InsightEngine:
    """
    Core analysis engine.

    Responsibilities:
      - Hold a reference to the LLM client (stub for now).
      - Inspect the dataframe and route questions to simple
        rule-based analyzers.
      - Optionally pass context + question to the LLM to
        refine or polish the answer later.
    """

    def __init__(self, llm_client: Optional[DeepSeekClient] = None) -> None:
        self.llm = llm_client or DeepSeekClient()

    # ---------- Public API ----------

    def analyze(
        self,
        df: pd.DataFrame,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> InsightResult:
        """
        Main entry point.

        df       : the uploaded dataframe
        question : user question in plain language
        history  : list of (role, text) if you want to keep conversation context
        """

        history = history or []
        q = (question or "").strip()

        if df.empty:
            return InsightResult(answer="The dataset is empty. There is nothing to analyze yet.")

        if not q:
            return self._overview(df)

        lower_q = q.lower()
        if any(w in lower_q for w in ["column", "columns", "fields", "schema", "structure"]):
            return self._overview(df)

        if any(w in lower_q for w in ["missing", "null", "na", "nan", "incomplete"]):
            return self._missingness(df, q)

        if any(w in lower_q for w in ["distribution", "histogram", "spread", "range"]):
            return self._distribution(df, q)

        if any(w in lower_q for w in ["trend", "over time", "time series", "per year", "per month"]):
            return self._time_trend(df, q)

        if any(w in lower_q for w in ["top", "most common", "frequency", "count", "group"]):
            return self._categorical_counts(df, q)

        return self._generic_answer(df, q, history)

    # ---------- Helpers ----------

    def _overview(self, df: pd.DataFrame) -> InsightResult:
        n_rows, n_cols = df.shape
        dtypes = df.dtypes.astype(str)

        lines = [
            f"This file has **{n_rows:,} rows** and **{n_cols} columns**.",
            "",
            "Column types:",
        ]
        for col, dt in dtypes.items():
            lines.append(f"- `{col}` → `{dt}`")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            sample = numeric_cols[:5]
            desc = df[sample].describe().round(3)

            text_stats = ["", "Basic summary for numeric columns:"]
            text_stats.append(desc.to_markdown())
            lines.extend(text_stats)

        return InsightResult(answer="\n".join(lines))

    def _missingness(self, df: pd.DataFrame, question: str) -> InsightResult:
        na_counts = df.isna().sum()
        na_pct = (na_counts / len(df)).round(3)

        table = (
            pd.DataFrame({"missing_count": na_counts, "missing_ratio": na_pct})
            .sort_values("missing_ratio", ascending=False)
        )

        text = [
            "Missing data by column (sorted by highest percentage):",
            "",
            table.to_markdown(),
        ]

        chart = (
            alt.Chart(
                table.reset_index().rename(columns={"index": "column"})
            )
            .mark_bar()
            .encode(
                x=alt.X("column:N", sort="-y", title="Column"),
                y=alt.Y("missing_ratio:Q", title="Missing ratio"),
                tooltip=["column", "missing_count", "missing_ratio"],
            )
            .properties(height=400)
        )

        return InsightResult(answer="\n".join(text), chart=chart, debug={"question": question})

    def _distribution(self, df: pd.DataFrame, question: str) -> InsightResult:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return InsightResult(answer="I do not see any numeric columns to build a distribution on.")

        target_col = self._pick_column_from_question(question, numeric_cols) or numeric_cols[0]

        series = df[target_col].dropna()
        desc = series.describe().round(3)

        lines = [
            f"Distribution for numeric column `{target_col}`:",
            "",
            desc.to_frame(name=target_col).to_markdown(),
        ]

        chart = (
            alt.Chart(pd.DataFrame({target_col: series}))
            .mark_bar()
            .encode(
                x=alt.X(f"{target_col}:Q", bin=alt.Bin(maxbins=30), title=target_col),
                y=alt.Y("count():Q", title="Count"),
            )
            .properties(height=400)
        )

        return InsightResult(answer="\n".join(lines), chart=chart)

    def _time_trend(self, df: pd.DataFrame, question: str) -> InsightResult:
        """
        Try to find a date-like column and a numeric column, then show trend.
        """
        date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

        if not date_cols:
            for col in df.columns:
                if any(word in col.lower() for word in ["date", "time", "year"]):
                    try:
                        parsed = pd.to_datetime(df[col], errors="raise")
                        df = df.copy()
                        df[col] = parsed
                        date_cols.append(col)
                        break
                    except Exception:
                        continue

        if not date_cols:
            return InsightResult(
                answer="I could not find a clear date or time column to build a trend. "
                       "Try asking for a distribution or summary instead."
            )

        date_col = date_cols[0]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return InsightResult(
                answer=f"I found a time column (`{date_col}`) but no numeric columns to aggregate."
            )

        target_col = self._pick_column_from_question(question, numeric_cols) or numeric_cols[0]

        df_trend = (
            df[[date_col, target_col]]
            .dropna()
            .groupby(date_col)[target_col]
            .mean()
            .reset_index()
        )

        lines = [
            f"Average `{target_col}` over time by `{date_col}`.",
            f"Rows used: {len(df_trend):,}",
        ]

        chart = (
            alt.Chart(df_trend)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{date_col}:T", title=date_col),
                y=alt.Y(f"{target_col}:Q", title=f"Average {target_col}"),
                tooltip=[date_col, target_col],
            )
            .properties(height=400)
        )

        return InsightResult(answer="\n".join(lines), chart=chart)

    def _categorical_counts(self, df: pd.DataFrame, question: str) -> InsightResult:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return InsightResult(answer="I do not see any clear categorical columns to count.")

        target_col = self._pick_column_from_question(question, cat_cols) or cat_cols[0]

        counts = (
            df[target_col]
            .fillna("(missing)")
            .value_counts()
            .reset_index()
            .rename(columns={target_col: "count", "index": target_col})
        )

        top = counts.head(20)

        lines = [
            f"Top values for `{target_col}` (up to 20 shown):",
            "",
            top.to_markdown(index=False),
        ]

        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y(f"{target_col}:N", sort="-x", title=target_col),
                tooltip=[target_col, "count"],
            )
            .properties(height=400)
        )

        return InsightResult(answer="\n".join(lines), chart=chart)

    def _generic_answer(
        self,
        df: pd.DataFrame,
        question: str,
        history: List[Tuple[str, str]],
    ) -> InsightResult:
        schema_lines = []
        for col, dt in df.dtypes.astype(str).items():
            schema_lines.append(f"- {col} ({dt})")

        sample = df.head(5).to_markdown(index=False)

        prompt = [
            "You are a data analyst.",
            "Here is the dataset schema:",
            *schema_lines,
            "",
            "Here are the first 5 rows:",
            sample,
            "",
            f"User question: {question}",
        ]
        prompt_text = "\n".join(prompt)

        messages = [{"role": "system", "content": "You explain data in clear, concrete language."}]
        for role, text in history[-4:]:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": prompt_text})

        llm_answer = self.llm.complete(messages)

        return InsightResult(
            answer=llm_answer,
            debug={"router": "generic", "question": question},
        )

    @staticmethod
    def _pick_column_from_question(question: str, candidates: List[str]) -> Optional[str]:
        """
        If the question mentions a column name (or part of it),
        try to use that one.
        """
        q = question.lower()
        best_match = None
        best_len = 0
        for col in candidates:
            name = col.lower()
            if name in q and len(name) > best_len:
                best_match = col
                best_len = len(name)
        return best_match







