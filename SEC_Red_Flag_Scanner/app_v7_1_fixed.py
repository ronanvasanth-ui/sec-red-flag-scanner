import os
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="SEC Filing Risk Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# APP CONFIG
# ============================================================

APP_NAME = "SEC Filing Risk Lab"
APP_VERSION = "7.1"
CACHE_TTL = 3600
REQUEST_TIMEOUT = 30
MAX_SEARCH_RESULTS = 30

# SEC asks automated users to identify themselves with a meaningful User-Agent.
# Set SEC_USER_AGENT in Streamlit secrets/environment for deployment.
def config_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return str(value).strip()
    try:
        secret_value = st.secrets.get(name, default)
        return str(secret_value).strip()
    except Exception:
        return default


SEC_USER_AGENT = config_value(
    "SEC_USER_AGENT",
    "SEC Filing Risk Lab/7.1 research app contact@example.com",
)

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

SEC_ARCHIVE_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


# ============================================================
# SEC HTTP / ERROR HANDLING
# ============================================================

class SECDataError(RuntimeError):
    """Readable exception for SEC/network/data problems."""


def sec_get(url: str, timeout: int = REQUEST_TIMEOUT, archive: bool = False):
    headers = SEC_ARCHIVE_HEADERS if archive else SEC_HEADERS
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SECDataError(f"SEC request failed: {exc}") from exc

    return response


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_tickers() -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = sec_get(url, timeout=20, archive=True)
        payload = response.json()
    except (ValueError, SECDataError) as exc:
        raise SECDataError(
            "Could not load the SEC company directory. Please retry in a moment."
        ) from exc

    rows = []
    for value in payload.values():
        try:
            rows.append(
                {
                    "ticker": str(value["ticker"]).upper().strip(),
                    "name": str(value["title"]).strip(),
                    "cik": str(int(value["cik_str"])).zfill(10),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    frame = pd.DataFrame(rows).drop_duplicates("ticker")
    if frame.empty:
        raise SECDataError("SEC company directory returned no usable companies.")
    return frame.sort_values("ticker").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_submissions(cik: str) -> dict:
    cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return sec_get(url).json()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_companyfacts(cik: str) -> dict:
    cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return sec_get(url).json()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_filing(cik: str, accession: str, document: str) -> Tuple[str, str]:
    accession_no_dashes = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_no_dashes}/{document}"
    )
    response = sec_get(url, timeout=45, archive=True)
    text = response.text
    # Filing text is used only for review-signal discovery, not numerical scoring.
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, url


# ============================================================
# FILING METADATA
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def annual_10ks(subs: dict) -> pd.DataFrame:
    recent = pd.DataFrame(subs.get("filings", {}).get("recent", {}))
    if recent.empty:
        return pd.DataFrame()

    required = {"form", "filingDate", "reportDate", "accessionNumber", "primaryDocument"}
    if not required.issubset(recent.columns):
        return pd.DataFrame()

    # Exclude amendments from the default annual-filing workflow. A 10-K/A can
    # be inspected later if needed, but using it silently would complicate
    # reproducibility.
    frame = recent[recent["form"].eq("10-K")].copy()
    if frame.empty:
        return frame

    frame["filingDate"] = pd.to_datetime(frame["filingDate"], errors="coerce")
    frame["reportDate"] = pd.to_datetime(frame["reportDate"], errors="coerce")
    frame = frame.dropna(subset=["filingDate", "reportDate"])
    frame = frame.sort_values(["reportDate", "filingDate", "accessionNumber"])
    frame = frame.drop_duplicates(subset=["reportDate"], keep="last")
    return frame.reset_index(drop=True)


def latest_10k(subs: dict) -> Optional[pd.Series]:
    filings = annual_10ks(subs)
    if filings.empty:
        return None
    return filings.iloc[-1]


# ============================================================
# XBRL CONCEPTS
# ============================================================

# Ordered alternatives. We use the first concept with a defensible observation
# for the target period, rather than summing overlapping concepts (which can
# double-count debt or other line items).
CONCEPTS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "ocf": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
        "AccountsNotesAndLoansReceivableNetCurrent",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings",
    ],
    "debt": [
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "assets": [
        "Assets",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}

DURATION_METRICS = {"revenue", "net_income", "ocf"}


def _parse_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts)


def _unit_for_concept(units: dict) -> Optional[str]:
    # Prefer USD; otherwise use the first available unit. We intentionally do
    # not mix currencies/units inside one metric.
    if "USD" in units:
        return "USD"
    preferred = [u for u in units if u.upper() in {"PURE", "SHARES"}]
    return preferred[0] if preferred else next(iter(units), None)


def _candidate_facts(
    facts: dict,
    metric: str,
    fiscal_year: int,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []

    for tag in CONCEPTS.get(metric, []):
        concept = usgaap.get(tag, {})
        units = concept.get("units", {})
        unit = _unit_for_concept(units)
        if not unit:
            continue

        for item in units.get(unit, []):
            form = str(item.get("form", ""))
            if form != "10-K" or "fy" not in item:
                continue

            try:
                fy = int(item["fy"])
                value = float(item["val"])
            except (TypeError, ValueError):
                continue

            if fy != int(fiscal_year):
                continue

            filed = _parse_timestamp(item.get("filed"))
            start = _parse_timestamp(item.get("start"))
            end = _parse_timestamp(item.get("end"))

            # Historical research must never use a fact first filed after the
            # filing being evaluated. This prevents future/restated information
            # from leaking backward into the backtest.
            if as_of is not None and filed is not None and filed > as_of:
                continue

            if metric in DURATION_METRICS:
                if start is None or end is None:
                    continue
                days = (end - start).days
                if not 300 <= days <= 380:
                    continue
            else:
                if end is None:
                    continue

            accession = str(item.get("accn", ""))
            rows.append(
                {
                    "tag": tag,
                    "unit": unit,
                    "fy": fy,
                    "start": start,
                    "end": end,
                    "filed": filed,
                    "accn": accession,
                    "value": value,
                }
            )

    return pd.DataFrame(rows)


def value_for_fy(
    facts: dict,
    metric: str,
    fiscal_year: int,
    as_of: Optional[pd.Timestamp] = None,
) -> Optional[float]:
    frame = _candidate_facts(facts, metric, fiscal_year, as_of=as_of)
    if frame.empty:
        return None

    # Prefer facts from the latest filing date available by the as-of boundary.
    # Then prefer the latest period end, making the choice deterministic.
    frame["filed_sort"] = frame["filed"].fillna(pd.Timestamp("1900-01-01"))
    frame["end_sort"] = frame["end"].fillna(pd.Timestamp("1900-01-01"))
    frame = frame.sort_values(["filed_sort", "end_sort", "accn", "tag"])
    return float(frame.iloc[-1]["value"])


def pct_growth(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def margin(value: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if value is None or denominator in (None, 0):
        return None
    return value / denominator * 100.0


# ============================================================
# QUANTITATIVE RISK FRAMEWORK
# ============================================================

SCORING_VERSION = "v7.1-rule-based"


def score_year(
    facts: dict,
    fy: int,
    as_of: Optional[pd.Timestamp] = None,
) -> Tuple[int, List[Tuple[str, str, int, str]], Dict[str, Optional[float]]]:
    metrics: Dict[str, Optional[float]] = {}
    flags = []
    points = 0

    rev0 = value_for_fy(facts, "revenue", fy - 1, as_of=as_of)
    rev1 = value_for_fy(facts, "revenue", fy, as_of=as_of)
    rec0 = value_for_fy(facts, "receivables", fy - 1, as_of=as_of)
    rec1 = value_for_fy(facts, "receivables", fy, as_of=as_of)
    ni1 = value_for_fy(facts, "net_income", fy, as_of=as_of)
    ocf1 = value_for_fy(facts, "ocf", fy, as_of=as_of)
    debt0 = value_for_fy(facts, "debt", fy - 1, as_of=as_of)
    debt1 = value_for_fy(facts, "debt", fy, as_of=as_of)
    inv0 = value_for_fy(facts, "inventory", fy - 1, as_of=as_of)
    inv1 = value_for_fy(facts, "inventory", fy, as_of=as_of)
    cash1 = value_for_fy(facts, "cash", fy, as_of=as_of)
    assets1 = value_for_fy(facts, "assets", fy, as_of=as_of)
    equity1 = value_for_fy(facts, "equity", fy, as_of=as_of)

    metrics["Revenue growth"] = pct_growth(rev0, rev1)
    metrics["Receivables growth"] = pct_growth(rec0, rec1)
    metrics["Debt growth"] = pct_growth(debt0, debt1)
    metrics["Inventory growth"] = pct_growth(inv0, inv1)
    metrics["OCF / net income"] = margin(ocf1, ni1)
    metrics["Cash / assets"] = margin(cash1, assets1)
    metrics["Equity / assets"] = margin(equity1, assets1)

    revenue_growth = metrics["Revenue growth"]
    receivables_growth = metrics["Receivables growth"]

    if revenue_growth is not None and receivables_growth is not None:
        gap = receivables_growth - revenue_growth
        metrics["Receivables minus revenue growth"] = gap
        if gap >= 15:
            points += 12
            flags.append(
                (
                    "Working capital",
                    "Receivables materially outpaced revenue",
                    12,
                    f"Gap: {gap:.1f} percentage points",
                )
            )

    if ni1 is not None and ocf1 is not None and ni1 > 0:
        ocf_ratio = ocf1 / ni1 * 100.0
        if ocf_ratio < 60:
            points += 12
            flags.append(
                (
                    "Cash flow",
                    "Operating cash flow materially trailed net income",
                    12,
                    f"OCF / net income: {ocf_ratio:.0f}%",
                )
            )

    debt_growth = metrics["Debt growth"]
    if debt_growth is not None and debt_growth >= 20:
        points += 8
        flags.append(
            (
                "Leverage",
                "Debt increased materially",
                8,
                f"Debt growth: {debt_growth:.1f}%",
            )
        )

    inventory_growth = metrics["Inventory growth"]
    if inventory_growth is not None and inventory_growth >= 25:
        points += 6
        flags.append(
            (
                "Business quality",
                "Inventory increased materially",
                6,
                f"Inventory growth: {inventory_growth:.1f}%",
            )
        )

    score = max(0, 100 - min(60, points))
    return score, flags, metrics


def risk_level(score: int) -> str:
    if score >= 85:
        return "Low"
    if score >= 70:
        return "Moderate"
    if score >= 50:
        return "Elevated"
    return "High"


# ============================================================
# QUALITATIVE FILING REVIEW
# ============================================================

RULES = [
    ("Accounting / controls", "Material weakness language", ["material weakness"], 3),
    ("Accounting / controls", "Restatement language", ["restatement", "restated financial statements"], 3),
    ("Liquidity", "Going-concern language", ["going concern", "substantial doubt about"], 4),
    ("Liquidity", "Debt covenant language", ["debt covenant", "covenant violation"], 2),
    ("Governance", "Related-party language", ["related party", "related-party"], 1),
    ("Business quality", "Customer concentration language", ["customer concentration", "concentration of customers"], 1),
    ("Business quality", "Impairment language", ["impairment charge"], 1),
    ("Business quality", "Restructuring language", ["restructuring charge"], 1),
]


def _sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def _context_supports_signal(sentence: str, label: str) -> bool:
    """Require evidence that the phrase describes an actual disclosed condition.

    This intentionally remains rule-based and transparent. It rejects common
    hypothetical/risk-factor boilerplate instead of treating any keyword hit
    as an event.
    """
    s = sentence.lower()
    if any(x in s for x in [
        "could result in", "could cause", "could adversely",
        "may result in", "may cause", "might result",
        "if we were to", "if such", "would result",
        "potential for", "risk of", "could have a material",
    ]):
        return False

    if label == "Material weakness language":
        return bool(re.search(
            r"\b(material weakness)\b.*\b(identified|exists|existed|concluded|determined|reported|disclosed)\b|"
            r"\b(identified|exists|existed|concluded|determined|reported|disclosed)\b.*\b(material weakness)\b",
            s,
        ))

    if label == "Restatement language":
        return bool(re.search(
            r"\b(restated|restatement)\b.*\b(financial statements|financial results|previously issued|prior period|periods)\b|"
            r"\b(financial statements|financial results|previously issued|prior period|periods)\b.*\b(restated|restatement)\b",
            s,
        ))

    if label == "Going-concern language":
        return bool(re.search(r"\b(going concern|substantial doubt about)\b", s))

    if label == "Debt covenant language":
        return bool(re.search(r"\b(covenant violation|debt covenant)\b", s))

    if label == "Impairment language":
        return bool(re.search(r"\bimpairment charge\b", s))

    if label == "Restructuring language":
        return bool(re.search(r"\brestructuring charge\b", s))

    # For lower-severity review signals, a direct contextual mention is useful
    # but is not treated as evidence of distress.
    return True


def filing_signals(text: str):
    sentences = _sentences(text)
    hits = []
    evidence = []

    for category, label, terms, severity in RULES:
        matched = None
        matched_sentence = None
        for sentence in sentences:
            lower_sentence = sentence.lower()
            term = next((term for term in terms if term in lower_sentence), None)
            if term and _context_supports_signal(sentence, label):
                matched = term
                matched_sentence = sentence
                break

        if matched is None:
            continue

        hits.append({
            "Category": category,
            "Signal": label,
            "Review severity": severity,
            "Trigger": matched,
        })
        evidence.append((label, matched_sentence[:900]))

    return hits, evidence[:10]


# ============================================================
# QUALITY / RESEARCH HELPERS
# ============================================================

SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "V", "MA", "WMT", "COST", "KO", "NFLX", "DIS",
    "ORCL", "PEP", "ADBE", "CRM", "INTC",
]

RESEARCH_SAMPLE_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "CSCO", "INTC", "QCOM", "AMD", "TXN", "AMAT", "MU", "IBM", "NOW",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
    "V", "MA", "PYPL", "COF", "SPGI", "CME", "ICE", "CB", "PGR", "ALL",
    "WMT", "COST", "HD", "LOW", "TGT", "TJX", "NKE", "MCD", "SBUX", "KO",
    "PEP", "PM", "MO", "CL", "PG", "EL", "KMB", "MDLZ", "GIS", "KHC",
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "CVS", "CI",
    "UNH", "ISRG", "ABT", "TMO", "DHR", "MDT", "SYK", "BSX", "GE", "CAT",
    "DE", "HON", "UPS", "RTX", "LMT", "BA", "UNP", "CSX", "ETN", "MMM",
    "XOM", "CVX", "COP", "SLB", "EOG", "NEE", "DUK", "SO", "T", "VZ",
]


def format_pct(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}%"


def correlation_label(value):
    if value is None or pd.isna(value):
        return "Not available"
    magnitude = abs(float(value))
    if magnitude < 0.10:
        strength = "negligible"
    elif magnitude < 0.30:
        strength = "weak"
    elif magnitude < 0.50:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "positive" if value > 0 else "negative" if value < 0 else "approximately zero"
    return f"{strength} {direction} association"


def score_range_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Score range"] = pd.cut(
        data["Score"],
        bins=[-1, 69, 84, 100],
        labels=["High risk (<70)", "Moderate (70–84)", "Low risk (85–100)"],
    )
    summary = (
        data.groupby("Score range", observed=False)["Forward revenue growth"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary.columns = [
        "Score range",
        "Observations",
        "Mean forward growth (%)",
        "Median forward growth (%)",
    ]
    return summary


def build_company_record(
    ticker: str,
    company: str,
    cik: str,
    facts: dict,
    filing: pd.Series,
    include_text: bool = True,
) -> dict:
    as_of = _parse_timestamp(filing["filingDate"])
    report_date = _parse_timestamp(filing["reportDate"])
    if report_date is None:
        raise SECDataError("Filing has no usable report date.")

    fy = int(report_date.year)
    score, qflags, metrics = score_year(facts, fy, as_of=as_of)

    review, evidence = [], []
    url = ""
    if include_text:
        try:
            text, url = get_filing(cik, filing["accessionNumber"], filing["primaryDocument"])
            review, evidence = filing_signals(text)
        except SECDataError:
            # Numerical analysis remains useful if the filing HTML is temporarily unavailable.
            pass

    data_quality = []
    for metric in [
        "Revenue growth",
        "Receivables growth",
        "Debt growth",
        "Inventory growth",
        "OCF / net income",
    ]:
        if metrics.get(metric) is None:
            data_quality.append(metric)

    return {
        "ticker": ticker,
        "company": company,
        "cik": cik,
        "fy": fy,
        "report_date": report_date,
        "filing_date": as_of,
        "accession": filing["accessionNumber"],
        "document": filing["primaryDocument"],
        "url": url,
        "score": score,
        "risk_level": risk_level(score),
        "qflags": qflags,
        "review": review,
        "evidence": evidence,
        "metrics": metrics,
        "missing_metrics": data_quality,
    }


# ============================================================
# HISTORICAL DATA
# ============================================================


def historical_observation(
    ticker: str,
    filing: pd.Series,
    facts: dict,
    next_filing: Optional[pd.Series] = None,
) -> Optional[dict]:
    as_of = _parse_timestamp(filing.get("filingDate"))
    report_date = _parse_timestamp(filing.get("reportDate"))
    if as_of is None or report_date is None:
        return None

    fy = int(report_date.year)
    current_rev = value_for_fy(facts, "revenue", fy, as_of=as_of)

    # Use the next year's first standard 10-K filing date as the information
    # boundary for the forward outcome. This avoids quietly using later
    # restatements when calculating the outcome variable.
    next_as_of = _parse_timestamp(next_filing.get("filingDate")) if next_filing is not None else None
    next_rev = value_for_fy(facts, "revenue", fy + 1, as_of=next_as_of)

    # Score strictly as-of the historical filing date. The forward outcome is
    # measured using information available by the following annual filing.
    score, flags, metrics = score_year(facts, fy, as_of=as_of)

    if current_rev is None or next_rev is None:
        return None

    return {
        "Ticker": ticker,
        "Fiscal year scored": fy,
        "Filing date": as_of.date().isoformat(),
        "Score": score,
        "Risk level": risk_level(score),
        "Quantitative flags": len(flags),
        "Forward revenue growth": pct_growth(current_rev, next_rev),
        "Revenue growth at score date": metrics.get("Revenue growth"),
        "Receivables-revenue gap": metrics.get("Receivables minus revenue growth"),
        "OCF / net income": metrics.get("OCF / net income"),
        "Debt growth": metrics.get("Debt growth"),
        "Inventory growth": metrics.get("Inventory growth"),
        "Scoring version": SCORING_VERSION,
    }


def run_backtest(
    tickers_df: pd.DataFrame,
    years_per_company: int = 6,
    selected_sample: Optional[List[str]] = None,
) -> pd.DataFrame:
    selected = selected_sample or SAMPLE
    rows: List[dict] = []
    progress = st.progress(0, text="Building historical observations…")

    for i, ticker in enumerate(selected):
        match = tickers_df[tickers_df["ticker"].eq(ticker)]
        if match.empty:
            continue

        company_row = match.iloc[0]
        try:
            subs = get_submissions(company_row["cik"])
            filings = annual_10ks(subs)
            facts = get_companyfacts(company_row["cik"])

            candidates = []
            filing_rows = list(filings.iterrows())
            for index, (_, filing) in enumerate(filing_rows):
                if _parse_timestamp(filing.get("reportDate")) is None:
                    continue
                next_filing = filing_rows[index + 1][1] if index + 1 < len(filing_rows) else None
                obs = historical_observation(ticker, filing, facts, next_filing=next_filing)
                if obs is not None:
                    candidates.append(obs)

            rows.extend(candidates[-int(max(1, years_per_company)):])
        except (SECDataError, KeyError, TypeError, ValueError):
            pass

        progress.progress(
            (i + 1) / len(selected),
            text=f"Processing {ticker} ({i + 1}/{len(selected)})…",
        )

    progress.empty()
    return pd.DataFrame(rows)


# ============================================================
# OPTIONAL PRIVACY-SAFE USAGE TELEMETRY
# ============================================================

# Set POSTHOG_API_KEY and POSTHOG_HOST to enable anonymous product analytics.
# Only event name + a salted anonymous session identifier are sent. No name,
# email, CIK, query text, or company-level selection is transmitted.
POSTHOG_API_KEY = config_value("POSTHOG_API_KEY")
POSTHOG_HOST = config_value("POSTHOG_HOST", "https://us.i.posthog.com").rstrip("/")


def anonymous_session_id() -> str:
    existing = st.session_state.get("anonymous_session_id")
    if existing:
        return existing
    raw = f"{datetime.utcnow().date().isoformat()}|{st.session_state.get('_anonymous_seed', os.urandom(16).hex())}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    st.session_state["anonymous_session_id"] = digest
    return digest


def track_event(event: str, properties: Optional[dict] = None) -> None:
    if not POSTHOG_API_KEY:
        return
    try:
        payload = {
            "api_key": POSTHOG_API_KEY,
            "event": event,
            "distinct_id": anonymous_session_id(),
            "properties": {
                "app_version": APP_VERSION,
                **(properties or {}),
            },
        }
        requests.post(
            f"{POSTHOG_HOST}/capture/",
            json=payload,
            timeout=3,
        )
    except requests.RequestException:
        pass


# ============================================================
# UI HELPERS
# ============================================================

def metric_strip(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def safe_download_df(frame: pd.DataFrame, filename: str, label: str):
    if frame is None or frame.empty:
        st.caption("Nothing to download yet.")
        return
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
        use_container_width=False,
    )


# ============================================================
# APP HEADER
# ============================================================

st.title("🔎 SEC Filing Risk Lab")
st.caption(
    "EDGAR + XBRL → transparent rule-based financial profile → filing review → historical research"
)
st.info(
    "Research tool, not investment advice. Quantitative scoring is intentionally separate from filing-language review. "
    "Historical tests use filing-date cutoffs for the scored year to reduce look-ahead leakage."
)

# First-load event. The session id is not persistent across browsers.
if "tracked_app_open" not in st.session_state:
    track_event("app_open")
    st.session_state["tracked_app_open"] = True

# ============================================================
# SIDEBAR / SETTINGS
# ============================================================

with st.sidebar:
    st.subheader("Research settings")
    backtest_years = st.slider(
        "Historical years per company",
        min_value=2,
        max_value=10,
        value=6,
        help="Uses the most recent eligible historical observations for each selected company.",
    )
    st.caption(f"Scoring methodology: `{SCORING_VERSION}`")
    st.markdown("---")
    st.markdown("**Data source**")
    st.write("SEC EDGAR submissions + standard XBRL company facts")
    st.caption("Automated access should use a meaningful SEC User-Agent and comply with SEC fair-access guidance.")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Company Scanner",
    "📊 Peer Research",
    "🧪 Historical Backtest",
    "📐 Methodology",
])


# ============================================================
# COMPANY SCANNER
# ============================================================

with tab1:
    st.subheader("Company Scanner")
    st.write(
        "Search a public company, score its latest 10-K using the transparent quantitative framework, "
        "and inspect separate filing-language review signals."
    )

    try:
        tickers = get_tickers()
    except SECDataError as exc:
        st.error(str(exc))
        st.stop()

    search = st.text_input(
        "Search by ticker or company name",
        placeholder="AAPL, NVIDIA, Microsoft…",
        key="company_search",
    ).strip()

    if search:
        upper = search.upper()
        mask = (
            tickers["ticker"].str.contains(re.escape(upper), regex=True, na=False)
            | tickers["name"].str.contains(re.escape(search), regex=True, case=False, na=False)
        )
        matches = tickers.loc[mask].head(MAX_SEARCH_RESULTS)
    else:
        matches = tickers.head(25)

    if matches.empty:
        st.warning("No matching companies found.")
        st.stop()

    option_map = {
        f"{row.ticker} — {row.name}": (row.ticker, row.cik, row.name)
        for _, row in matches.iterrows()
    }

    selection = st.selectbox("Company", list(option_map.keys()), key="company_selection")
    ticker, cik, company_name = option_map[selection]

    if st.button("Run latest 10-K scan", type="primary", use_container_width=True):
        with st.spinner(f"Reading {ticker}'s latest 10-K and XBRL data…"):
            try:
                subs = get_submissions(cik)
                filing = latest_10k(subs)
                if filing is None:
                    raise SECDataError("No standard 10-K was found in the available filing history.")
                facts = get_companyfacts(cik)
                result = build_company_record(ticker, company_name, cik, facts, filing)
                st.session_state["result"] = result
                track_event("company_scan_run")
            except SECDataError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected scan error: {type(exc).__name__}: {exc}")

    if "result" in st.session_state:
        result = st.session_state["result"]
        metrics = result["metrics"]

        st.markdown("---")
        metric_strip([
            ("Financial profile score", f"{result['score']}/100", "Higher is stronger under this rule set."),
            ("Risk band", result["risk_level"], "Rule-based interpretation; not a probability of distress."),
            ("Quantitative flags", str(len(result["qflags"])), "Flags triggered by predefined accounting thresholds."),
            ("Review signals", str(len(result["review"])), "Keyword-based filing signals shown separately from the score."),
        ])

        st.caption(
            f"FY {result['fy']} · Report date {result['report_date'].date()} · Filed {result['filing_date'].date()} · "
            f"Scoring version {SCORING_VERSION}"
        )

        if result["missing_metrics"]:
            st.warning(
                "Some supporting metrics were unavailable from standard XBRL concepts: "
                + ", ".join(result["missing_metrics"])
                + ". Missing data are not treated as negative signals."
            )

        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Quantitative risk signals")
            if result["qflags"]:
                qframe = pd.DataFrame(
                    [
                        {
                            "Category": category,
                            "Signal": signal,
                            "Points": points,
                            "Evidence": evidence,
                        }
                        for category, signal, points, evidence in result["qflags"]
                    ]
                )
                st.dataframe(qframe, use_container_width=True, hide_index=True)
            else:
                st.success("No predefined quantitative thresholds were triggered.")

            st.subheader("Core financial metrics")
            metric_frame = pd.DataFrame(
                [
                    {"Metric": key, "Value": format_pct(value)}
                    for key, value in metrics.items()
                ]
            )
            st.dataframe(metric_frame, use_container_width=True, hide_index=True)

        with right:
            st.subheader("Filing-language review")
            if result["review"]:
                st.dataframe(pd.DataFrame(result["review"]), use_container_width=True, hide_index=True)
                st.caption("These are search-based review prompts, not conclusions. Read the cited context in the filing.")
            else:
                st.success("No predefined review phrases were detected.")

        st.subheader("Evidence from the filing")
        if result["evidence"]:
            for index, (label, snippet) in enumerate(result["evidence"], start=1):
                with st.expander(f"{index}. {label}"):
                    st.write(snippet)
        else:
            st.caption("No evidence snippets available because no filing text was retrieved.")

        st.subheader("Suggested analyst follow-ups")
        followups = [
            "Reconcile net income with operating cash flow and working-capital movements.",
            "Investigate receivables and inventory growth relative to revenue.",
            "Read debt maturities, covenant disclosures, and liquidity sources in context.",
            "Check relevant footnotes and risk-factor disclosures before drawing conclusions.",
        ]
        for item in followups:
            st.write(f"• {item}")

        if result["url"]:
            st.link_button("Open SEC filing", result["url"], use_container_width=False)


# ============================================================
# PEER RESEARCH
# ============================================================

with tab2:
    st.subheader("Peer Research")
    st.write(
        "Run the same scoring system across a fixed, reproducible sample. The default set is deliberately unchanged "
        "from v6 so longitudinal comparisons remain possible."
    )

    selected_peers = st.multiselect(
        "Sample companies",
        options=sorted(set(SAMPLE) | set(tickers["ticker"].head(50).tolist())),
        default=SAMPLE,
        key="peer_sample",
        help="Use a fixed sample for research reproducibility. Avoid hand-picking after seeing results when doing a study.",
    )

    if not selected_peers:
        st.warning("Select at least one company.")
    elif st.button("Run peer study", type="primary", use_container_width=True):
        progress = st.progress(0, text="Analyzing peer sample…")
        rows = []
        try:
            for i, peer in enumerate(selected_peers):
                match = tickers[tickers["ticker"].eq(peer)]
                if match.empty:
                    continue
                row = match.iloc[0]
                try:
                    subs = get_submissions(row["cik"])
                    filing = latest_10k(subs)
                    if filing is None:
                        continue
                    facts = get_companyfacts(row["cik"])
                    record = build_company_record(peer, row["name"], row["cik"], facts, filing, include_text=False)
                    rows.append(
                        {
                            "Ticker": peer,
                            "Company": row["name"],
                            "Fiscal year": record["fy"],
                            "Risk score": record["score"],
                            "Risk level": record["risk_level"],
                            "Quantitative flags": len(record["qflags"]),
                            "Revenue growth": record["metrics"].get("Revenue growth"),
                            "Debt growth": record["metrics"].get("Debt growth"),
                            "OCF / net income": record["metrics"].get("OCF / net income"),
                        }
                    )
                except (SECDataError, KeyError, TypeError, ValueError):
                    continue
                progress.progress((i + 1) / len(selected_peers), text=f"Processing {peer} ({i + 1}/{len(selected_peers)})…")
            progress.empty()
            study = pd.DataFrame(rows)
            st.session_state["study"] = study
            track_event("peer_study_run")
        except Exception as exc:
            progress.empty()
            st.error(f"Peer study failed: {type(exc).__name__}: {exc}")

    study = st.session_state.get("study", pd.DataFrame())
    if not study.empty:
        display = study.sort_values(["Risk score", "Ticker"], ascending=[True, True]).reset_index(drop=True)
        metric_strip([
            ("Companies analyzed", str(len(display)), "Number of companies with usable latest 10-K data."),
            ("Average score", f"{display['Risk score'].mean():.1f}/100", "Arithmetic mean of the rule-based score."),
            ("Below 70", str(int((display['Risk score'] < 70).sum())), "Companies in the elevated/high bands under this framework."),
        ])
        st.dataframe(display.style.format({
            "Revenue growth": "{:.1f}%",
            "Debt growth": "{:.1f}%",
            "OCF / net income": "{:.0f}%",
        }), use_container_width=True, hide_index=True)
        st.bar_chart(display.set_index("Ticker")["Risk score"])
        safe_download_df(display, "sec_risk_lab_peer_research.csv", "Download peer dataset")

        if display["Risk score"].nunique() <= 4:
            st.warning("The score has little variation in this sample, so ranking differences should not be over-interpreted.")
        else:
            st.info("This is descriptive cross-sectional research, not proof that one company is a better investment than another.")


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

with tab3:
    st.subheader("Historical Backtest")
    st.write(
        "The core test is: **does an earlier, filing-date-only score relate to the following year's reported revenue growth?** "
        "This is a simple falsifiable test, not a trading strategy."
    )
    st.warning(
        "Important research design: the scored year's facts are restricted to facts filed on or before that year's 10-K filing date. "
        "The following year's revenue is intentionally used only as the forward outcome. This reduces look-ahead leakage from later filings/restatements."
    )

    sample_preset = st.radio(
        "Research sample",
        ["20-company pilot", "100-company research sample", "Custom"],
        horizontal=True,
        key="backtest_sample_preset",
        help="Use the 20-company pilot for quick checks; use the 100-company sample for the final research run.",
    )
    if sample_preset == "20-company pilot":
        default_backtest = SAMPLE
    elif sample_preset == "100-company research sample":
        default_backtest = RESEARCH_SAMPLE_100
    else:
        default_backtest = SAMPLE

    backtest_options = sorted(set(SAMPLE) | set(RESEARCH_SAMPLE_100) | set(tickers["ticker"].head(100).tolist()))
    backtest_sample = st.multiselect(
        "Backtest sample",
        options=backtest_options,
        default=default_backtest,
        key="backtest_sample",
    )

    if backtest_sample and st.button("Run historical backtest", type="primary", use_container_width=True):
        try:
            bt = run_backtest(tickers, years_per_company=backtest_years, selected_sample=backtest_sample)
            st.session_state["backtest"] = bt
            track_event("backtest_run")
        except Exception as exc:
            st.error(f"Backtest failed: {type(exc).__name__}: {exc}")

    bt = st.session_state.get("backtest", pd.DataFrame()).copy()
    if not bt.empty:
        bt = bt.dropna(subset=["Score", "Forward revenue growth"]).reset_index(drop=True)
        st.metric("Historical observations", len(bt))
        st.caption("Each row is a company-year observation. Scores are calculated using information available by that historical filing date.")

        st.dataframe(
            bt.style.format({
                "Forward revenue growth": "{:.1f}%",
                "Revenue growth at score date": "{:.1f}%",
                "Receivables-revenue gap": "{:.1f}%",
                "OCF / net income": "{:.0f}%",
                "Debt growth": "{:.1f}%",
                "Inventory growth": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        if len(bt) >= 5 and bt["Score"].nunique() >= 2:
            pearson = bt["Score"].corr(bt["Forward revenue growth"], method="pearson")
            spearman = bt["Score"].rank(method="average").corr(bt["Forward revenue growth"].rank(method="average"))
            metric_strip([
                ("Pearson", "N/A" if pd.isna(pearson) else f"{pearson:.2f}", "Linear association."),
                ("Spearman ρ", "N/A" if pd.isna(spearman) else f"{spearman:.2f}", "Rank-based association."),
                ("Unique scores", str(bt["Score"].nunique()), "More variation generally makes discrimination tests more informative."),
            ])

            st.info(
                f"Observed relationship: Pearson r = {pearson:.2f} ({correlation_label(pearson)}). "
                "This does not establish causation, investment performance, or generalizable predictive power."
            )

            st.scatter_chart(
                bt,
                x="Score",
                y="Forward revenue growth",
                x_label="Historical framework score",
                y_label="Following-year revenue growth (%)",
            )

            summary = score_range_summary(bt)
            st.subheader("Subsequent growth by score band")
            st.dataframe(
                summary.style.format({
                    "Mean forward growth (%)": "{:.1f}%",
                    "Median forward growth (%)": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

            largest = int(summary["Observations"].max()) if not summary.empty else 0
            if len(bt) and largest > 0.75 * len(bt):
                st.warning("One score band contains more than 75% of observations; band comparisons are therefore unstable.")

            # Per-signal incidence across the historical sample.
            st.subheader("Signal incidence")
            signal_rows = []
            for label, threshold in [
                ("Receivables growth outpaced revenue by ≥15pp", "Receivables-revenue gap"),
                ("Debt growth ≥20%", "Debt growth"),
                ("Inventory growth ≥25%", "Inventory growth"),
                ("OCF / net income <60%", "OCF / net income"),
            ]:
                series = bt[threshold]
                if threshold == "Receivables-revenue gap":
                    triggered = series.ge(15).fillna(False)
                elif threshold == "Debt growth":
                    triggered = series.ge(20).fillna(False)
                elif threshold == "Inventory growth":
                    triggered = series.ge(25).fillna(False)
                else:
                    triggered = series.lt(60).fillna(False)
                signal_rows.append(
                    {
                        "Signal": label,
                        "Available observations": int(series.notna().sum()),
                        "Triggered": int(triggered.sum()),
                        "Trigger rate": (triggered.sum() / series.notna().sum() * 100) if series.notna().sum() else np.nan,
                    }
                )
            signal_df = pd.DataFrame(signal_rows)
            st.dataframe(signal_df.style.format({"Trigger rate": "{:.1f}%"}), use_container_width=True, hide_index=True)

        else:
            st.info("Not enough observations/score variation for a meaningful correlation display. Expand the sample or historical window.")

        safe_download_df(bt, "sec_risk_lab_historical_backtest.csv", "Download historical dataset")


# ============================================================
# METHODOLOGY
# ============================================================

with tab4:
    st.subheader("Methodology & Research Notes")
    st.markdown(
        f"**Scoring version:** `{SCORING_VERSION}`\n\n"
        "The quantitative score starts at 100 and subtracts points for four predefined accounting patterns. "
        "The score is a screening framework, not a calibrated default probability or investment recommendation."
    )

    st.markdown("### Quantitative scoring")
    scoring_table = pd.DataFrame([
        ["Receivables growth ≥ revenue growth + 15pp", 12, "Working-capital quality"],
        ["Operating cash flow < 60% of positive net income", 12, "Cash conversion"],
        ["Debt growth ≥ 20%", 8, "Leverage"],
        ["Inventory growth ≥ 25%", 6, "Inventory buildup"],
    ], columns=["Rule", "Penalty", "Interpretation"])
    st.dataframe(scoring_table, use_container_width=True, hide_index=True)

    st.markdown("### XBRL selection safeguards")
    for item in [
        "Duration metrics (revenue, net income, operating cash flow) require an approximately annual reporting period.",
        "Standard 10-K facts are preferred; 10-K/A amendments are not silently substituted into the default workflow.",
        "Historical scores use only facts whose `filed` date is on or before the historical 10-K filing date.",
        "Missing XBRL concepts remain missing; the framework does not convert unavailable data into negative signals.",
        "Debt concepts are treated as ordered alternatives rather than automatically summed, reducing overlap/double-counting risk.",
    ]:
        st.write(f"• {item}")

    st.markdown("### Filing-language review")
    st.write(
        "The filing scanner searches for a small, transparent dictionary of review phrases and checks sentence context before displaying a signal. "
        "Hypothetical risk-factor boilerplate is filtered out for the highest-severity accounting/control signals. These signals are displayed separately "
        "because even contextual language does not establish distress or wrongdoing."
    )

    st.markdown("### Historical backtest")
    st.write(
        "The backtest compares a filing-date-constrained score from fiscal year t with reported revenue growth from t to t+1. "
        "This tests association with one forward accounting outcome. It does not test stock returns, fraud prediction, causality, "
        "or whether the framework would have generated an investable trading strategy."
    )

    st.markdown("### Reproducibility checklist")
    for item in [
        "Keep the sample definition fixed before inspecting results.",
        "Record the scoring version whenever publishing a result.",
        "Do not tune thresholds on the same observations used for final evaluation.",
        "Prefer a held-out time period for any future research-grade validation.",
        "Report missing-data rates and failed company observations alongside headline results.",
    ]:
        st.write(f"• {item}")

    st.markdown("### Limitations")
    st.write(
        "SEC XBRL data are standardized but not perfectly uniform across issuers. Companies can use different concepts, "
        "presentation structures, fiscal calendars, and disclosures. The framework therefore emphasizes transparency and "
        "comparability over complexity. Any serious research use should validate observations against the underlying filing."
    )

    st.markdown("### Usage analytics")
    if POSTHOG_API_KEY:
        st.success("Anonymous product analytics are enabled on this deployment.")
        st.caption("Only an anonymous session identifier and generic event names are sent; company selections and search text are not transmitted.")
    else:
        st.info("Usage analytics are disabled. Set POSTHOG_API_KEY to enable anonymous event tracking for product research.")
