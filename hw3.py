#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [1, 5, 6]


ANALYSIS_QUERY = """
WITH annual_revenue AS (
    SELECT substr(order_date, 1, 4) AS year, SUM(amount) AS revenue
    FROM orders
    WHERE order_date >= '2023-01-01' AND order_date < '2026-01-01'
    GROUP BY year
),
annual_costs AS (
    SELECT substr(month, 1, 4) AS year,
           SUM(cost_of_sales + selling_admin + research_development) AS total_cost
    FROM monthly_costs
    WHERE month >= '2023-01' AND month < '2026-01'
    GROUP BY year
),
pivot AS (
    SELECT
        (SELECT revenue FROM annual_revenue WHERE year = '2023') AS revenue_2023,
        (SELECT revenue FROM annual_revenue WHERE year = '2024') AS revenue_2024,
        (SELECT revenue FROM annual_revenue WHERE year = '2025') AS revenue_2025,
        (SELECT total_cost FROM annual_costs WHERE year = '2024') AS total_cost_2024,
        (SELECT total_cost FROM annual_costs WHERE year = '2025') AS total_cost_2025
)
SELECT c.name, c.industry,
       m.revenue_growth_2y_pct AS median_growth,
       m.operating_margin_pct AS median_margin,
       p.*
FROM company c
JOIN industry_medians m ON m.industry = c.industry
CROSS JOIN pivot p
"""


def _one_sql_row(result: str) -> dict[str, str]:
    """Parse the single aggregate row returned by the provided sql_query tool."""
    if result.startswith("SQL error:"):
        raise RuntimeError(result)
    lines = [line for line in result.splitlines() if line.strip()]
    if len(lines) < 2 or lines[1] == "(no rows)":
        raise RuntimeError("The company metrics query returned no data.")
    headers = [part.strip() for part in lines[0].split("|")]
    values = [part.strip() for part in lines[1].split("|")]
    return dict(zip(headers, values))


def _calculate_metrics(tools: list[Any]) -> dict[str, Any]:
    """Collect evidence through the supplied tool and calculate all ratios in Python."""
    row = _one_sql_row(tools[2].invoke({"query": ANALYSIS_QUERY}))
    numbers = {
        key: float(value)
        for key, value in row.items()
        if key not in {"name", "industry"} and value not in (None, "", "None")
    }
    required = {
        "median_growth", "median_margin", "revenue_2023", "revenue_2024",
        "revenue_2025", "total_cost_2024", "total_cost_2025",
    }
    missing = required - numbers.keys()
    if missing:
        raise RuntimeError(f"Missing required metrics: {', '.join(sorted(missing))}")
    if numbers["revenue_2023"] <= 0 or numbers["revenue_2024"] <= 0 or numbers["revenue_2025"] <= 0:
        raise RuntimeError("Annual revenue must be positive to calculate growth and margins.")

    growth = (numbers["revenue_2025"] / numbers["revenue_2023"] - 1.0) * 100.0
    margin_2024 = (numbers["revenue_2024"] - numbers["total_cost_2024"]) / numbers["revenue_2024"] * 100.0
    margin_2025 = (numbers["revenue_2025"] - numbers["total_cost_2025"]) / numbers["revenue_2025"] * 100.0
    return {
        "company": row.get("name", "The company"),
        "industry": row.get("industry", "its industry"),
        "growth_2y_pct": growth,
        "median_growth_pct": numbers["median_growth"],
        "margin_2024_pct": margin_2024,
        "margin_2025_pct": margin_2025,
        "median_margin_pct": numbers["median_margin"],
        "margin_change_pp": margin_2025 - margin_2024,
    }


def _rule_verdicts(metrics: dict[str, Any]) -> tuple[dict[int, str], str]:
    """Apply fixed Fisher rules; the language model never decides these verdicts."""
    points = {
        1: "PASS" if metrics["growth_2y_pct"] >= metrics["median_growth_pct"] else "FAIL",
        5: "PASS" if metrics["margin_2025_pct"] >= metrics["median_margin_pct"] else "FAIL",
        6: "PASS" if metrics["margin_change_pp"] >= 0 else "FAIL",
    }
    summary = (
        f"Point 1 {points[1]}: FY2023-FY2025 revenue growth was {metrics['growth_2y_pct']:.1f}%, "
        f"versus the {metrics['industry']} industry median of {metrics['median_growth_pct']:.1f}%. "
        f"Point 5 {points[5]}: FY2025 operating margin was {metrics['margin_2025_pct']:.1f}%, "
        f"versus the industry median of {metrics['median_margin_pct']:.1f}%. "
        f"Point 6 {points[6]}: operating margin changed by {metrics['margin_change_pp']:+.1f} "
        f"percentage points from FY2024 ({metrics['margin_2024_pct']:.1f}%) "
        f"to FY2025 ({metrics['margin_2025_pct']:.1f}%)."
    )
    return points, summary


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _valid_model_summary(summary: str, points: dict[int, str]) -> bool:
    """Accept only a complete, self-consistent summary that covers all three points."""
    if not summary or len(summary) < 40 or summary[-1] not in ".!?":
        return False
    if "analysis failed" in summary.lower() or re.search(r"\d+\.\d{2,}", summary):
        return False
    for point, verdict in points.items():
        match = re.search(
            rf"\bpoint\s*{point}\b[^.!?]{{0,80}}?\b(PASS|FAIL)\b",
            summary,
            flags=re.IGNORECASE,
        )
        if match is None or match.group(1).upper() != verdict:
            return False
    return True


def build_agent() -> Any:
    """Create the required DeepSeek model once; deterministic rules remain in Python."""
    import os

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or "PASTE_" in key.upper() or "YOUR_" in key.upper():
        return None
    try:
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=1600,
            timeout=45,
            max_retries=2,
        )
    except Exception as exc:
        print(f"DeepSeek model could not be initialized; deterministic summaries will be used: {exc}")
        return None


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Run your agent and return its verdicts on CHOSEN_POINTS for every company.

    ``companies`` maps each company folder name to its four LangChain tools:

        list_documents()             names and dates of the company's documents
        read_document(name)          the text of one document
        sql_query(query)             one read-only SQLite query on company.db
        interview(role, topic)       one interview answer (INTERVIEW_LIMIT per company)

    Give the tools to a tool-calling agent, e.g.
    ``create_agent(model, tools=companies[name], system_prompt=...)``, or call
    them yourself with ``tool.invoke({...})``. The README explains the
    documents, the database tables and the interview roles and topics.

    Return one dictionary per company folder name, e.g. with
    CHOSEN_POINTS = [1, 7, 15]:

        {
            "hollins_group": {
                "points": {1: "PASS", 7: "FAIL", 15: "PASS"},
                "summary": "Two or three sentences explaining the verdicts.",
            },
            ...
        }

    Each chosen point needs a verdict ("PASS" or "FAIL"; True/False also
    works). Other points are ignored.

    Catch errors per company (e.g. a failed API call) and still return an
    entry for it: an exception here means no results.csv, which scores zero.
    You may analyse companies in parallel (e.g. ``asyncio.gather``).
    """
    results: dict[str, dict[str, Any]] = {}
    for name, tools in companies.items():
        try:
            metrics = _calculate_metrics(tools)
            points, fallback_summary = _rule_verdicts(metrics)
            summary = fallback_summary
            if agent is not None:
                display_metrics = {
                    key: round(value, 1) if isinstance(value, float) else value
                    for key, value in metrics.items()
                }
                evidence = json.dumps({"display_metrics": display_metrics, "verdicts": points}, ensure_ascii=False)
                prompt = (
                    "Write one concise English paragraph of at most three sentences summarizing this "
                    "Fisher analysis for an investment report. Preserve every supplied PASS/FAIL verdict "
                    "exactly and mention the rounded evidence for points 1, 5 and 6. Use the supplied "
                    "display values exactly; do not calculate new values, use more precise decimals, "
                    "add outside facts, use Markdown, or output JSON. Evidence: " + evidence
                )
                try:
                    response = await asyncio.wait_for(agent.ainvoke(prompt), timeout=60)
                    model_summary = _response_text(response)
                    if _valid_model_summary(model_summary, points):
                        summary = model_summary
                    elif model_summary:
                        print(f"{name}: incomplete model summary; using deterministic summary")
                except Exception as exc:
                    print(f"{name}: summary model failed; using deterministic summary: {exc}")
            results[name] = {"points": points, "summary": summary}
        except Exception as exc:
            results[name] = {
                "points": {},
                "summary": f"Analysis failed for this company: {type(exc).__name__}: {exc}",
            }
    return results


# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())
