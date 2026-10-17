"""Offline logic tests for SEC Filing Risk Lab v7.2.

The application is a Streamlit script, so these tests compile selected pure helper
functions from its AST rather than launching the UI or making SEC/network requests.
"""
import ast
from pathlib import Path
import re
import html
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP = Path(__file__).with_name("app.py")
TREE = ast.parse(APP.read_text())
WANTED = {
    "maximum_drawdown", "summarize_flag_outcomes", "drawdown_tail_summary",
    "extract_filing_section", "extract_10k_sections", "compare_disclosure_sections",
    "_canonical_custom_match", "score_year", "pct_growth", "margin", "_sentences",
    "_cluster_bootstrap_median_difference", "_parse_timestamp", "_unit_for_concept", "_candidate_facts",
}
NODES = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in WANTED]
NS = {
    "np": np, "pd": pd, "re": re, "unescape": html.unescape,
    "TfidfVectorizer": TfidfVectorizer, "cosine_similarity": cosine_similarity,
    "Optional": __import__("typing").Optional, "List": __import__("typing").List,
    "Dict": __import__("typing").Dict, "Tuple": __import__("typing").Tuple,
    "SCORING_VERSION": "test",
    "CONCEPTS": {
        "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        "net_income": ["NetIncomeLoss", "ProfitLoss"],
        "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
        "receivables": ["AccountsReceivableNetCurrent", "AccountsReceivableNet", "AccountsNotesAndLoansReceivableNetCurrent"],
        "inventory": ["InventoryNet", "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings"],
        "debt": ["LongTermDebtAndFinanceLeaseObligationsNoncurrent", "LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtCurrent"],
        "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
        "assets": ["Assets"], "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    },
    "DURATION_METRICS": {"revenue", "net_income", "ocf"},
}
# The real Streamlit decorator is not needed for these helper tests; the selected
# functions do not use decorated fetchers.
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(APP), "exec"), NS)


def test_maximum_drawdown_uses_peak_to_trough():
    prices = pd.Series([100, 120, 90, 95] * 63)
    value = NS["maximum_drawdown"](prices, trading_days=252)
    assert value == -25.0


def test_maximum_drawdown_rejects_incomplete_window():
    assert NS["maximum_drawdown"](pd.Series([100, 90] * 40), trading_days=252) is None


def test_flag_outcomes_compare_flagged_and_unflagged():
    frame = pd.DataFrame({
        "Receivables flag": [True, True, False, False, True, False],
        "Debt flag": [False, True, False, False, True, False],
        "Cash-flow flag": [False, False, True, False, False, False],
        "Inventory flag": [False, False, False, True, False, False],
        "Forward revenue growth": [1, 3, 10, 12, 5, 8],
    })
    out = NS["summarize_flag_outcomes"](frame)
    row = out[out["Signal"] == "Receivables flag"].iloc[0]
    assert row["Flagged n"] == 3
    assert row["Unflagged n"] == 3
    assert row["Median difference (flagged − unflagged)"] < 0


def test_custom_tag_mapping_requires_unit_period_and_statement_alignment():
    metric, status, _ = NS["_canonical_custom_match"](
        "Total Revenue", "EntityTotalRevenue", "USD", "duration", ["StatementOfIncome"]
    )
    assert metric == "revenue"
    assert status == "High-confidence candidate; eligible for fallback"
    metric, status, reason = NS["_canonical_custom_match"](
        "Total Revenue", "EntityTotalRevenue", "EUR", "duration", ["StatementOfIncome"]
    )
    assert metric == "revenue"
    assert "Excluded" in status and "unit" in reason
    metric, status, _ = NS["_canonical_custom_match"](
        "Revenue from Subscription Services", "EntitySubscriptionRevenue", "USD", "duration", ["StatementOfIncome"]
    )
    assert metric is None and "component concept" in status


def test_receivables_threshold_is_strictly_more_than_20_percentage_points():
    # Provide a tiny controlled `value_for_fy` double so score_year can be
    # evaluated without network data or a full SEC CompanyFacts fixture.
    values = {
        ("revenue", 2023): 100.0, ("revenue", 2024): 110.0,
        ("receivables", 2023): 100.0, ("receivables", 2024): 130.0,
        ("net_income", 2024): 100.0, ("ocf", 2024): 90.0,
        ("debt", 2023): 100.0, ("debt", 2024): 100.0,
        ("inventory", 2023): 100.0, ("inventory", 2024): 100.0,
        ("cash", 2024): 100.0, ("assets", 2023): 1000.0,
        ("assets", 2024): 1000.0, ("equity", 2024): 500.0,
    }
    NS["value_for_fy"] = lambda facts, metric, year, as_of=None: values.get((metric, year))
    _, flags, metrics = NS["score_year"]({}, 2024, as_of=pd.Timestamp("2025-02-01"))
    assert metrics["Receivables minus revenue growth"] == 20.0
    assert not any(item[1] == "Receivables materially outpaced revenue" for item in flags)
    values[("receivables", 2024)] = 131.0
    _, flags, metrics = NS["score_year"]({}, 2024, as_of=pd.Timestamp("2025-02-01"))
    assert metrics["Receivables minus revenue growth"] > 20.0
    assert any(item[1] == "Receivables materially outpaced revenue" for item in flags)


def test_disclosure_similarity_detects_a_change_and_returns_terms():
    prior_text = ("The company sells enterprise software under annual contracts. " * 100)
    current_text = ("The company sells enterprise software under annual contracts and expanded payment terms. " * 100)
    prev = {"Risk Factors": prior_text, "MD&A": prior_text}
    cur = {"Risk Factors": current_text, "MD&A": current_text}
    result = NS["compare_disclosure_sections"](prev, cur)
    assert result["Risk Factors"]["available"]
    assert 0 <= result["Risk Factors"]["cosine_similarity"] <= 1
    assert result["Risk Factors"]["cosine_similarity"] > 0.5
    assert not result["Risk Factors"]["changes"].empty


def test_drawdown_tail_summary_uses_more_negative_quartile():
    n = 16
    data = pd.DataFrame({
        "Forward 252-day max drawdown (%)": [-50, -45, -40, -35, -30, -25, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2],
        "Receivables flag": [True] * 4 + [False] * 12,
        "Debt flag": [True] * 5 + [False] * 11,
        "Cash-flow flag": [False] * n,
        "Inventory flag": [False] * n,
    })
    out = NS["drawdown_tail_summary"](data)
    row = out[out["Signal"] == "Receivables flag"].iloc[0]
    assert row["Flag rate in worst quartile (%)"] > row["Flag rate elsewhere (%)"]


def test_custom_fact_is_a_fallback_only_when_standard_metric_is_missing():
    custom = {
        "facts": {"us-gaap": {}},
        "_normalized_custom": {
            "revenue": [{
                "tag": "custom:EntityTotalRevenue", "unit": "USD", "fy": 2024,
                "start": pd.Timestamp("2024-01-01"), "end": pd.Timestamp("2024-12-31"),
                "filed": pd.Timestamp("2025-02-15"), "accn": "custom-accn", "value": 150.0,
            }]
        },
    }
    before_filing = NS["_candidate_facts"](custom, "revenue", 2024, as_of=pd.Timestamp("2025-02-14"))
    after_filing = NS["_candidate_facts"](custom, "revenue", 2024, as_of=pd.Timestamp("2025-02-16"))
    assert before_filing.empty  # no look-ahead through the extension fallback
    assert len(after_filing) == 1 and after_filing.iloc[0]["value"] == 150.0

    standard = {
        "facts": {"us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [{
                    "form": "10-K", "fy": 2024, "val": 200.0,
                    "filed": "2025-02-10", "start": "2024-01-01", "end": "2024-12-31", "accn": "std-accn",
                }]}
            }
        }},
        "_normalized_custom": custom["_normalized_custom"],
    }
    chosen = NS["_candidate_facts"](standard, "revenue", 2024, as_of=pd.Timestamp("2025-02-16"))
    assert len(chosen) == 1 and chosen.iloc[0]["value"] == 200.0
    assert not str(chosen.iloc[0]["tag"]).startswith("custom:")
