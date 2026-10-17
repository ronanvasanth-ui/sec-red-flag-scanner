import os
import re
import hashlib
from datetime import datetime
from html import unescape
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional research dependencies. The core SEC scanner still runs without them.
try:
    import yfinance as yf
except ImportError:  # pragma: no cover - depends on deployment environment
    yf = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

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
APP_VERSION = "7.2"
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
    "SEC Filing Risk Lab/7.2 research app contact@example.com",
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
def get_filing_html(cik: str, accession: str, document: str) -> Tuple[str, str]:
    """Fetch the raw filing HTML so section headings remain available for parsing."""
    accession_no_dashes = str(accession).replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_no_dashes}/{document}"
    )
    response = sec_get(url, timeout=45, archive=True)
    return response.text, url


def strip_html(raw_html: str) -> str:
    """Convert filing HTML to searchable plain text without external dependencies."""
    if BeautifulSoup is not None:
        text = BeautifulSoup(raw_html, "html.parser").get_text(" ")
    else:
        text = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_filing(cik: str, accession: str, document: str) -> Tuple[str, str]:
    raw_html, url = get_filing_html(cik, accession, document)
    # Filing text is used for transparent review prompts, not numerical scoring.
    return strip_html(raw_html), url


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

    # Use a custom extension only when no usable standard US-GAAP observation
    # exists for this metric and period. Extension mappings are conservative:
    # ambiguous concepts are excluded rather than silently scored.
    if not rows:
        custom_facts = facts.get("_normalized_custom", {}).get(metric, [])
        for item in custom_facts:
            try:
                if int(item.get("fy")) != int(fiscal_year):
                    continue
                filed = _parse_timestamp(item.get("filed"))
                start = _parse_timestamp(item.get("start"))
                end = _parse_timestamp(item.get("end"))
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            if as_of is not None and filed is not None and filed > as_of:
                continue
            if metric in DURATION_METRICS:
                if start is None or end is None or not 300 <= (end - start).days <= 380:
                    continue
            elif end is None:
                continue
            rows.append({
                "tag": str(item.get("tag", "custom:unknown")),
                "unit": str(item.get("unit", "USD")),
                "fy": int(fiscal_year),
                "start": start,
                "end": end,
                "filed": filed,
                "accn": str(item.get("accn", "")),
                "value": value,
            })
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

SCORING_VERSION = "v7.2-rule-based"


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

    assets0 = value_for_fy(facts, "assets", fy - 1, as_of=as_of)
    average_assets = None
    if assets0 is not None and assets1 is not None and assets0 > 0 and assets1 > 0:
        average_assets = (assets0 + assets1) / 2.0
    if ni1 is not None and ocf1 is not None and average_assets not in (None, 0):
        # Simple balance-sheet-scaled proxy, not a full accrual-accounting measure.
        metrics["Accruals proxy / average assets"] = (ni1 - ocf1) / average_assets * 100.0
    else:
        metrics["Accruals proxy / average assets"] = None

    revenue_growth = metrics["Revenue growth"]
    receivables_growth = metrics["Receivables growth"]

    if revenue_growth is not None and receivables_growth is not None:
        gap = receivables_growth - revenue_growth
        metrics["Receivables minus revenue growth"] = gap
        if gap > 20:
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


CONTEXT_SEARCH_GROUPS = {
    "Payment / contract terms": [
        r"payment terms", r"extended payment", r"enterprise contract", r"contractual payment",
        r"billing terms", r"customer terms", r"payment period", r"days sales outstanding",
    ],
    "Receivables / collections": [
        r"accounts receivable", r"trade receivables", r"receivables were collected",
        r"subsequent collection", r"collections of receivables", r"cash receipts",
        r"collection period", r"past due receivables",
    ],
    "Accruals / cash conversion": [
        r"accrued revenue", r"accrued receivables", r"noncash", r"working capital",
        r"operating cash flow", r"cash conversion",
    ],
}


def filing_context_passages(text: str, max_per_group: int = 4) -> List[dict]:
    """Surface filing sentences worth human review; a keyword hit is not proof."""
    sentences = _sentences(text)
    output = []
    for group, patterns in CONTEXT_SEARCH_GROUPS.items():
        count = 0
        for sentence in sentences:
            if any(re.search(pattern, sentence, flags=re.I) for pattern in patterns):
                output.append({"Topic": group, "Passage": sentence[:1200]})
                count += 1
                if count >= max_per_group:
                    break
    return output


def search_next_two_10q_collection_evidence(cik: str, submissions: dict, after_date,
                                             max_filings: int = 2) -> List[dict]:
    """Search the next two filed 10-Qs for receivables/collection passages.

    This surfaces evidence to inspect; it cannot establish how much of a specific
    receivables balance was collected unless the filing says so explicitly.
    """
    recent = pd.DataFrame(submissions.get("filings", {}).get("recent", {}))
    required = {"form", "filingDate", "accessionNumber", "primaryDocument"}
    if recent.empty or not required.issubset(recent.columns):
        return []
    recent["filingDate"] = pd.to_datetime(recent["filingDate"], errors="coerce")
    cutoff = _parse_timestamp(after_date)
    if cutoff is None:
        return []
    quarters = recent[(recent["form"] == "10-Q") & (recent["filingDate"] > cutoff)].dropna(subset=["filingDate"])
    quarters = quarters.sort_values("filingDate").head(max_filings)
    output = []
    patterns = re.compile(r"receivabl|collect(?:ed|ion|ions|ing)?|cash receipts|invoice|past due|payment terms", re.I)
    for _, filing in quarters.iterrows():
        try:
            text, url = get_filing(cik, filing["accessionNumber"], filing["primaryDocument"])
        except SECDataError:
            continue
        passages = [sentence[:1200] for sentence in _sentences(text) if patterns.search(sentence)]
        # Limit noisy keyword returns; retain multiple snippets so users can inspect context.
        for passage in passages[:8]:
            output.append({
                "Filing date": filing["filingDate"].date().isoformat(),
                "Form": "10-Q",
                "Passage": passage,
                "SEC filing": url,
            })
    return output


def extract_filing_section(plain_text: str, start_pattern: str, end_pattern: str,
                           max_chars: int = 180_000) -> str:
    """Extract the longest plausible section between repeated filing headings.

    10-K tables of contents repeat headings, so the longest reasonable start/end
    pair is preferred over the first occurrence. The output should be spot-checked.
    """
    starts = list(re.finditer(start_pattern, plain_text, flags=re.I))
    ends = list(re.finditer(end_pattern, plain_text, flags=re.I))
    candidates = []
    for start in starts:
        for end in ends:
            if end.start() <= start.end():
                continue
            length = end.start() - start.end()
            if 700 <= length <= max_chars:
                candidates.append((length, plain_text[start.end():end.start()].strip()))
            break
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


def extract_10k_sections(raw_html: str) -> Dict[str, str]:
    plain = strip_html(raw_html)
    patterns = {
        "Risk Factors": (
            r"\bitem\s*1a\s*[.\-:–—]?\s*risk factors\b",
            r"\bitem\s*1b\b",
        ),
        "MD&A": (
            r"\bitem\s*7\s*[.\-:–—]?\s*management.{0,100}?discussion.{0,80}?analysis\b",
            r"\bitem\s*7a\b",
        ),
    }
    return {name: extract_filing_section(plain, start, end) for name, (start, end) in patterns.items()}


def compare_disclosure_sections(previous: Dict[str, str], current: Dict[str, str]) -> Dict[str, dict]:
    """Compare same-company consecutive 10-K sections using TF-IDF/cosine similarity."""
    results = {}
    if TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError("Disclosure comparison requires scikit-learn. Install it from requirements_v7_2.txt.")
    for section in ("Risk Factors", "MD&A"):
        old_text = previous.get(section, "")
        new_text = current.get(section, "")
        if len(old_text) < 700 or len(new_text) < 700:
            results[section] = {"available": False, "reason": "Could not reliably extract both sections; verify filing format or section headings."}
            continue
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=7000,
                                     min_df=1, max_df=1.0, sublinear_tf=True)
        matrix = vectorizer.fit_transform([old_text[:180_000], new_text[:180_000]])
        similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
        terms = vectorizer.get_feature_names_out()
        delta = (matrix[1] - matrix[0]).toarray().ravel()
        current_weights = matrix[1].toarray().ravel()
        old_weights = matrix[0].toarray().ravel()
        order = np.argsort(np.abs(delta))[::-1]
        changes = []
        for idx in order:
            if abs(float(delta[idx])) < 1e-6:
                continue
            changes.append({
                "Term / phrase": str(terms[idx]),
                "Change in TF-IDF weight": float(delta[idx]),
                "Prior weight": float(old_weights[idx]),
                "Current weight": float(current_weights[idx]),
            })
            if len(changes) >= 20:
                break
        def sentence_for_term(text: str, term: str) -> str:
            for sentence in _sentences(text):
                if re.search(re.escape(term), sentence, flags=re.I):
                    return sentence[:900]
            return "No single sentence matched this term; inspect the full section."

        evidence_rows = []
        for change in changes[:8]:
            term = change["Term / phrase"]
            evidence_rows.append({
                "Term / phrase": term,
                "Prior-filing passage": sentence_for_term(old_text, term),
                "Current-filing passage": sentence_for_term(new_text, term),
            })
        results[section] = {
            "available": True,
            "cosine_similarity": similarity,
            "cosine_distance": 1.0 - similarity,
            "previous_chars": len(old_text),
            "current_chars": len(new_text),
            "changes": pd.DataFrame(changes),
            "evidence_passages": pd.DataFrame(evidence_rows),
        }
    return results


# ============================================================
# CUSTOM XBRL EXTENSION NORMALIZATION
# ============================================================

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
XLINK_LABEL = "{http://www.w3.org/1999/xlink}label"
XLINK_FROM = "{http://www.w3.org/1999/xlink}from"
XLINK_TO = "{http://www.w3.org/1999/xlink}to"
XLINK_ROLE = "{http://www.w3.org/1999/xlink}role"


def _local_name(tag: str) -> str:
    return str(tag).split("}")[-1]


def _xml_text(element, name: str) -> Optional[str]:
    for child in element.iter():
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_filing_index(cik: str, accession: str) -> dict:
    accession_no_dashes = str(accession).replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/index.json"
    return sec_get(url, timeout=30, archive=True).json()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_archive_xml(cik: str, accession: str, filename: str) -> str:
    accession_no_dashes = str(accession).replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{filename}"
    return sec_get(url, timeout=35, archive=True).text


def _extension_labels(label_xml: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    try:
        root = ET.fromstring(label_xml)
    except ET.ParseError:
        return labels
    for link in root.iter():
        if _local_name(link.tag) != "labelLink":
            continue
        locs, resources, arcs = {}, {}, []
        for child in list(link):
            kind = _local_name(child.tag)
            if kind == "loc":
                locs[child.attrib.get(XLINK_LABEL, "")] = child.attrib.get(XLINK_HREF, "").split("#")[-1]
            elif kind == "label":
                resources[child.attrib.get(XLINK_LABEL, "")] = unescape("".join(child.itertext())).strip()
            elif kind == "labelArc":
                arcs.append((child.attrib.get(XLINK_FROM, ""), child.attrib.get(XLINK_TO, "")))
        for source, target in arcs:
            fragment = locs.get(source, "")
            label = resources.get(target, "")
            if fragment and label:
                local = fragment.split("_")[-1]
                labels.setdefault(local, label)
    return labels


def _extension_presentation_roles(presentation_xml: str) -> Dict[str, List[str]]:
    roles: Dict[str, List[str]] = {}
    try:
        root = ET.fromstring(presentation_xml)
    except ET.ParseError:
        return roles
    for link in root.iter():
        if _local_name(link.tag) != "presentationLink":
            continue
        role_uri = link.attrib.get(XLINK_ROLE, "")
        role_hint = role_uri.rsplit("/", 1)[-1].replace("_", " ")
        locs = {}
        for child in list(link):
            if _local_name(child.tag) == "loc":
                locs[child.attrib.get(XLINK_LABEL, "")] = child.attrib.get(XLINK_HREF, "").split("#")[-1].split("_")[-1]
        for child in list(link):
            if _local_name(child.tag) == "presentationArc":
                concept = locs.get(child.attrib.get(XLINK_TO, ""), "")
                if concept:
                    roles.setdefault(concept, []).append(role_hint)
    return roles


def _canonical_custom_match(label: str, local_tag: str, unit: str, period_type: str,
                             statement_roles: List[str]) -> Tuple[Optional[str], str, str]:
    """Conservative extension-tag mapping using label, unit, period and statement role."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", f"{label} {local_tag}").lower()
    words = re.sub(r"[_:]+", " ", words)
    roles = " ".join(statement_roles).lower()
    # Do not map obvious component/segment tags to consolidated metrics.
    if re.search(r"\b(allowance|reserve|provision|bad debt|impairment|past due|overdue)\b", words) and re.search(r"receivable|inventory", words):
        return None, "Excluded — component concept", "Label/tag describes a reserve, allowance or subset rather than a clear consolidated balance."
    revenue_qualifiers = r"\b(subscription|advertising|product|services?|segment|license|licensing|cloud|hardware|software|domestic|international|retail|wholesale)\b"
    is_canonical_contract_revenue = bool(re.search(r"\brevenue from contracts? with customers?\b", words))
    if re.search(revenue_qualifiers, words) and not is_canonical_contract_revenue and re.search(r"revenue|sales", words):
        return None, "Excluded — component concept", "Label/tag appears to describe a revenue component or segment, not consolidated revenue."
    if re.search(r"\b(attributable to noncontrolling|discontinued operations|continuing operations)\b", words) and re.search(r"net income|net earnings|profit", words):
        return None, "Excluded — component concept", "Label/tag describes a subset of net income rather than a comparable consolidated measure."
    metric = None
    expected_period = None
    strong_label = False
    if re.search(r"\b(net sales|sales revenue|revenue from|total revenue|revenues?)\b", words) and not re.search(r"deferred revenue|unearned revenue", words):
        metric, expected_period, strong_label = "revenue", "duration", True
    elif re.search(r"\b(net income|net earnings|profit loss|profit for the year|net profit)\b", words):
        metric, expected_period, strong_label = "net_income", "duration", True
    elif re.search(r"\b(net cash provided by operating|cash provided by operating|net cash from operating|operating cash flows)\b", words):
        metric, expected_period, strong_label = "ocf", "duration", True
    elif re.search(r"\b(accounts receivable|account receivable|trade receivable|receivables)\b", words):
        metric, expected_period, strong_label = "receivables", "instant", True
    elif re.search(r"\binventory\b", words) and not re.search(r"inventory based compensation", words):
        metric, expected_period, strong_label = "inventory", "instant", True
    elif re.search(r"\b(long term debt|short term debt|borrowings|debt outstanding|total debt)\b", words):
        metric, expected_period, strong_label = "debt", "instant", True
    elif re.search(r"\b(cash and cash equivalents|cash equivalents)\b", words):
        metric, expected_period, strong_label = "cash", "instant", True
    elif re.search(r"\b(total assets|assets total)\b", words):
        metric, expected_period, strong_label = "assets", "instant", True
    elif re.search(r"\b(stockholders equity|shareholders equity|total equity)\b", words):
        metric, expected_period, strong_label = "equity", "instant", True

    if metric is None:
        return None, "Excluded — no defensible canonical match", "Label/tag did not match a supported metric pattern."
    issues = []
    if unit.upper() != "USD":
        issues.append(f"unit is {unit or 'unknown'}, not USD")
    if period_type != expected_period:
        issues.append(f"period is {period_type or 'unknown'}, expected {expected_period}")
    expected_roles = {
        "revenue": ("income", "operation", "profit", "earnings"),
        "net_income": ("income", "operation", "profit", "earnings"),
        "ocf": ("cash",),
        "receivables": ("balance", "position", "asset"),
        "inventory": ("balance", "position", "asset"),
        "debt": ("balance", "position", "liabilit"),
        "cash": ("balance", "position", "asset"),
        "assets": ("balance", "position"),
        "equity": ("balance", "position", "equity"),
    }
    if roles and not any(token in roles for token in expected_roles.get(metric, ())):
        issues.append("presentation role does not clearly match the expected statement")
    if not roles:
        issues.append("presentation role unavailable")
    if issues:
        return metric, "Excluded — ambiguous / incompatible", "; ".join(issues)
    if strong_label:
        return metric, "High-confidence candidate; eligible for fallback", "Label, USD unit, period type and presentation role align."
    return metric, "Excluded — ambiguous / incompatible", "Concept match is not sufficiently specific."


def extract_custom_xbrl_observations(cik: str, filing: pd.Series) -> Tuple[List[dict], pd.DataFrame]:
    """Inspect custom extension facts in a filing; conservatively map supported concepts.

    Successful mappings are used only as fallbacks when a usable standard US-GAAP fact
    is absent. All other candidate facts remain visible in an audit table and are excluded.
    """
    accession = str(filing.get("accessionNumber", ""))
    filed_date = _parse_timestamp(filing.get("filingDate"))
    if not accession or filed_date is None:
        return [], pd.DataFrame()
    index_data = get_filing_index(cik, accession)
    files = index_data.get("directory", {}).get("item", [])
    if not files:
        files = index_data.get("files", [])
    def file_name(item):
        return str(item.get("name", item.get("filename", "")))
    def file_type(item):
        return str(item.get("type", item.get("fileType", ""))).upper()
    instance_item = next((it for it in files if "INS" in file_type(it) and file_name(it).lower().endswith(".xml")), None)
    if instance_item is None:
        instance_item = next((it for it in files if file_name(it).lower().endswith("_htm.xml")), None)
    if instance_item is None:
        return [], pd.DataFrame([{"Tag": "", "Status": "Unavailable", "Reason": "No XBRL instance document found in filing index."}])
    lab_item = next((it for it in files if "LAB" in file_type(it) and file_name(it).lower().endswith(".xml")), None)
    pre_item = next((it for it in files if "PRE" in file_type(it) and file_name(it).lower().endswith(".xml")), None)
    instance_xml = get_archive_xml(cik, accession, file_name(instance_item))
    label_map = {}
    role_map = {}
    if lab_item is not None:
        label_map = _extension_labels(get_archive_xml(cik, accession, file_name(lab_item)))
    if pre_item is not None:
        role_map = _extension_presentation_roles(get_archive_xml(cik, accession, file_name(pre_item)))
    try:
        root = ET.fromstring(instance_xml)
    except ET.ParseError as exc:
        raise SECDataError(f"Could not parse the filing's XBRL instance XML: {exc}") from exc

    core_fragments = ("fasb.org/us-gaap", "fasb.org/srt", "fasb.org/dei", "xbrl.org/", "xbrl.us/", "xbrl.sec.gov/", "sec.gov/Archives/edgar/data")
    contexts, units = {}, {}
    for element in root.iter():
        if _local_name(element.tag) == "context":
            ident = element.attrib.get("id", "")
            start = _xml_text(element, "startDate")
            end = _xml_text(element, "endDate")
            instant = _xml_text(element, "instant")
            has_dimensions = any(_local_name(desc.tag) in {"explicitMember", "typedMember"} for desc in element.iter())
            contexts[ident] = {"start": start, "end": end or instant, "period_type": "duration" if start and end else "instant" if instant else "unknown", "has_dimensions": has_dimensions}
        elif _local_name(element.tag) == "unit":
            ident = element.attrib.get("id", "")
            measures = [str(child.text or "").rsplit(":", 1)[-1] for child in element.iter() if _local_name(child.tag) == "measure"]
            units[ident] = "USD" if "USD" in measures else (measures[0] if measures else "")

    normalized, audit = [], []
    for element in root.iter():
        if "}" not in element.tag or element.text is None:
            continue
        namespace, local = element.tag[1:].split("}", 1)
        if any(fragment in namespace.lower() for fragment in core_fragments):
            continue
        if local.lower() in {"schemaref", "context", "unit", "footnote", "footnotearc"}:
            continue
        context_ref = element.attrib.get("contextRef")
        unit_ref = element.attrib.get("unitRef")
        if not context_ref or context_ref not in contexts:
            continue
        try:
            value = float(element.text.strip().replace(",", ""))
        except (ValueError, AttributeError):
            continue
        context = contexts[context_ref]
        # Avoid dimensioned segment facts (e.g., revenue for one product line)
        # from being treated as consolidated company totals.
        if context.get("has_dimensions"):
            continue
        label = label_map.get(local, re.sub(r"([a-z])([A-Z])", r"\1 \2", local).replace("_", " "))
        roles = role_map.get(local, [])
        metric, status, rationale = _canonical_custom_match(label, local, units.get(unit_ref, ""), context["period_type"], roles)
        relevant_tag = metric is not None or re.search(r"revenue|sales|receivable|inventory|borrow|debt|cash|asset|equity|profit|income", f"{label} {local}", re.I)
        if relevant_tag:
            audit.append({
            "Tag": local,
            "Label": label,
            "Candidate metric": metric or "—",
            "Unit": units.get(unit_ref, ""),
            "Period type": context["period_type"],
            "Statement context": "; ".join(roles[:3]) or "Unavailable",
            "Value": value,
            "Status": status,
            "Reason": rationale,
            })
        if status == "High-confidence candidate; eligible for fallback" and metric:
            start = _parse_timestamp(context.get("start"))
            end = _parse_timestamp(context.get("end"))
            if end is None:
                continue
            normalized.append({
                "metric": metric,
                "tag": f"custom:{local}",
                "unit": "USD",
                "fy": int(end.year),
                "start": start,
                "end": end,
                "filed": filed_date,
                "accn": accession,
                "value": value,
            })
    # If multiple distinct extension tags map to the same metric/date in one filing,
    # do not choose arbitrarily. Mark the competing candidates excluded.
    from collections import defaultdict
    candidate_groups = defaultdict(list)
    for item in normalized:
        group_key = (item["metric"], item["fy"], str(item["end"]), str(item["filed"]))
        candidate_groups[group_key].append(item)
    ambiguous_keys = {key for key, items in candidate_groups.items() if len({item["tag"] for item in items}) > 1}
    if ambiguous_keys:
        keep = []
        for item in normalized:
            group_key = (item["metric"], item["fy"], str(item["end"]), str(item["filed"]))
            if group_key not in ambiguous_keys:
                keep.append(item)
        normalized = keep
        for row in audit:
            if row.get("Status") != "High-confidence candidate; eligible for fallback":
                continue
            for metric in [row.get("Candidate metric")]:
                group_key = (metric, int(pd.to_datetime(row.get("Period end")).year), str(pd.to_datetime(row.get("Period end"))), str(filed_date)) if row.get("Period end") else None
                if group_key in ambiguous_keys:
                    row["Status"] = "Excluded — multiple custom tags map to same metric/period"
                    row["Reason"] = "Competing extension concepts share a metric and period; no arbitrary selection was made."
                    break
    return normalized, pd.DataFrame(audit)


def merge_normalized_custom_facts(facts: dict, observations: List[dict]) -> dict:
    """Return a shallow-copied CompanyFacts object with audited custom fallbacks."""
    out = dict(facts)
    normalized = {key: list(value) for key, value in facts.get("_normalized_custom", {}).items()}
    for observation in observations:
        metric = observation.get("metric")
        if metric:
            item = {key: value for key, value in observation.items() if key != "metric"}
            normalized.setdefault(metric, []).append(item)
    out["_normalized_custom"] = normalized
    return out


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

# A predetermined holdout set that does not overlap with the original 20-company pilot.
# It is a reproducible *new* holdout, not evidence that a prior essay experiment used these names.
HOLDOUT_SAMPLE_10 = ["AVGO", "CSCO", "QCOM", "AMD", "TXN", "AMAT", "MU", "IBM", "BAC", "WFC"]


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


def _cluster_bootstrap_median_difference(data: pd.DataFrame, flag: str, outcome: str,
                                         seed: int = 20261017, repetitions: int = 400) -> Tuple[float, float]:
    """95% cluster-bootstrap interval for flagged-minus-unflagged median difference."""
    if "Ticker" not in data.columns or data["Ticker"].nunique() < 5:
        return np.nan, np.nan
    clusters = data["Ticker"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    estimates = []
    grouped = {ticker: data[data["Ticker"] == ticker] for ticker in clusters}
    for _ in range(repetitions):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[ticker] for ticker in sampled_clusters], ignore_index=True)
        flagged = sample.loc[sample[flag].astype(bool), outcome].dropna()
        unflagged = sample.loc[~sample[flag].astype(bool), outcome].dropna()
        if len(flagged) and len(unflagged):
            estimates.append(float(flagged.median() - unflagged.median()))
    if len(estimates) < max(40, repetitions // 5):
        return np.nan, np.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize_flag_outcomes(frame: pd.DataFrame, outcome: str = "Forward revenue growth") -> pd.DataFrame:
    """Flagged/unflagged statistics and cluster-bootstrap intervals where feasible."""
    flag_columns = [
        "Receivables flag", "Debt flag", "Cash-flow flag", "Inventory flag",
    ]
    rows = []
    if frame.empty or outcome not in frame.columns:
        return pd.DataFrame()
    for flag_index, flag in enumerate(flag_columns):
        if flag not in frame.columns:
            continue
        needed = [flag, outcome] + (["Ticker"] if "Ticker" in frame.columns else [])
        data = frame[needed].dropna(subset=[outcome]).copy()
        data = data[data[flag].notna()]
        flagged = data[data[flag].astype(bool)][outcome].astype(float)
        unflagged = data[~data[flag].astype(bool)][outcome].astype(float)
        pearson = spearman = np.nan
        if len(data) >= 5 and data[flag].nunique() > 1 and data[outcome].nunique() > 1:
            pearson = data[flag].astype(int).corr(data[outcome].astype(float), method="pearson")
            spearman = data[flag].astype(int).corr(data[outcome].astype(float), method="spearman")
        ci_low, ci_high = _cluster_bootstrap_median_difference(
            data, flag, outcome, seed=20261017 + flag_index
        )
        rows.append({
            "Signal": flag,
            "Outcome": outcome,
            "Flagged n": int(len(flagged)),
            "Unflagged n": int(len(unflagged)),
            "Flagged median": float(flagged.median()) if len(flagged) else np.nan,
            "Unflagged median": float(unflagged.median()) if len(unflagged) else np.nan,
            "Median difference (flagged − unflagged)": float(flagged.median() - unflagged.median()) if len(flagged) and len(unflagged) else np.nan,
            "Flagged mean": float(flagged.mean()) if len(flagged) else np.nan,
            "Unflagged mean": float(unflagged.mean()) if len(unflagged) else np.nan,
            "Mean difference (flagged − unflagged)": float(flagged.mean() - unflagged.mean()) if len(flagged) and len(unflagged) else np.nan,
            "Pearson r (binary flag)": float(pearson) if pd.notna(pearson) else np.nan,
            "Spearman ρ (binary flag)": float(spearman) if pd.notna(spearman) else np.nan,
            "Median-difference 95% CI low (cluster bootstrap)": ci_low,
            "Median-difference 95% CI high (cluster bootstrap)": ci_high,
        })
    return pd.DataFrame(rows)


def maximum_drawdown(prices: pd.Series, trading_days: int = 252) -> Optional[float]:
    """Return max peak-to-trough drawdown (%) over the first 252 post-signal closes."""
    if prices is None:
        return None
    series = pd.to_numeric(prices, errors="coerce").dropna()
    if isinstance(series.index, pd.DatetimeIndex):
        series = series.sort_index()
    series = series[series > 0]
    if len(series) < int(trading_days):
        return None
    series = series.iloc[:int(trading_days)]
    drawdowns = series / series.cummax() - 1.0
    if drawdowns.empty:
        return None
    return float(drawdowns.min() * 100.0)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_adjusted_close(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch adjusted daily closes. Requires optional yfinance; end_date is exclusive."""
    if yf is None:
        raise RuntimeError("Stock-price drawdown testing requires yfinance. Install requirements_v7_2.txt.")
    raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True,
                      progress=False, threads=False, actions=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        for col in raw.columns:
            if str(col[0]).lower() == "close":
                close = raw[col]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                break
    elif "Close" in raw.columns:
        close = raw["Close"]
    if close is None:
        return pd.Series(dtype=float)
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) is not None else pd.to_datetime(close.index)
    return close.sort_index()


def attach_drawdowns(backtest: pd.DataFrame) -> pd.DataFrame:
    """Attach post-filing max drawdown to each row; incomplete one-year windows stay missing."""
    data = backtest.copy()
    if data.empty:
        data["Forward 252-day max drawdown (%)"] = pd.Series(dtype=float)
        return data
    data["Forward 252-day max drawdown (%)"] = np.nan
    parsed_dates = pd.to_datetime(data["Filing date"], errors="coerce")
    data["_filing_dt"] = parsed_dates
    valid = data.dropna(subset=["_filing_dt"])
    for ticker, group in valid.groupby("Ticker"):
        earliest = group["_filing_dt"].min() + pd.Timedelta(days=1)
        latest = group["_filing_dt"].max() + pd.Timedelta(days=390)
        try:
            prices = get_adjusted_close(str(ticker), earliest.date().isoformat(), latest.date().isoformat())
        except Exception as exc:
            data.loc[group.index, "Drawdown data status"] = f"Unavailable: {type(exc).__name__}"
            continue
        for idx, row in group.iterrows():
            filing_date = row["_filing_dt"]
            window = prices[prices.index > filing_date].iloc[:252]
            drawdown = maximum_drawdown(window, trading_days=252)
            data.loc[idx, "Forward 252-day max drawdown (%)"] = drawdown if drawdown is not None else np.nan
            data.loc[idx, "Drawdown data status"] = "Complete" if drawdown is not None else "Incomplete / unavailable 252-day window"
    data.drop(columns=["_filing_dt"], inplace=True)
    return data


def drawdown_tail_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare signal prevalence in the worst drawdown quartile vs all other observations."""
    outcome = "Forward 252-day max drawdown (%)"
    if outcome not in frame.columns:
        return pd.DataFrame()
    data = frame.dropna(subset=[outcome]).copy()
    if len(data) < 8:
        return pd.DataFrame()
    cutoff = float(data[outcome].quantile(0.25))
    data["Worst drawdown quartile"] = data[outcome] <= cutoff
    rows = []
    for flag in ["Receivables flag", "Debt flag", "Cash-flow flag", "Inventory flag"]:
        if flag not in data.columns:
            continue
        available = data[data[flag].notna()]
        tail = available[available["Worst drawdown quartile"]]
        other = available[~available["Worst drawdown quartile"]]
        rows.append({
            "Signal": flag,
            "Worst-quartile n": int(len(tail)),
            "Flag rate in worst quartile (%)": float(tail[flag].astype(bool).mean() * 100) if len(tail) else np.nan,
            "Other observations n": int(len(other)),
            "Flag rate elsewhere (%)": float(other[flag].astype(bool).mean() * 100) if len(other) else np.nan,
            "Difference (percentage points)": (float(tail[flag].astype(bool).mean() - other[flag].astype(bool).mean()) * 100) if len(tail) and len(other) else np.nan,
            "Worst-quartile cutoff (%)": cutoff,
        })
    return pd.DataFrame(rows)


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

    review, evidence, context_passages = [], [], []
    url = ""
    if include_text:
        try:
            text, url = get_filing(cik, filing["accessionNumber"], filing["primaryDocument"])
            review, evidence = filing_signals(text)
            context_passages = filing_context_passages(text)
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
        "context_passages": context_passages,
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
    next_rev = value_for_fy(facts, "revenue", fy + 1, as_of=next_as_of) if next_as_of is not None else None

    # Score strictly as-of the historical filing date. The forward outcome is
    # measured using information available by the following annual filing.
    score, flags, metrics = score_year(facts, fy, as_of=as_of)
    flag_names = {item[1] for item in flags}

    # Deliberately flawed comparison score: allows the latest/restated facts in
    # CompanyFacts. This exists only to demonstrate look-ahead sensitivity.
    naive_score, _, _ = score_year(facts, fy, as_of=None)

    if current_rev is None or next_rev is None:
        return None

    return {
        "Ticker": ticker,
        "Fiscal year scored": fy,
        "Filing date": as_of.date().isoformat(),
        "Accession number": str(filing.get("accessionNumber", "")),
        "Primary document": str(filing.get("primaryDocument", "")),
        "Score": score,
        "Naive score (invalid look-ahead diagnostic)": naive_score,
        "Risk level": risk_level(score),
        "Quantitative flags": len(flags),
        "Receivables flag": "Receivables materially outpaced revenue" in flag_names,
        "Debt flag": "Debt increased materially" in flag_names,
        "Cash-flow flag": "Operating cash flow materially trailed net income" in flag_names,
        "Inventory flag": "Inventory increased materially" in flag_names,
        "Forward revenue growth": pct_growth(current_rev, next_rev),
        "Revenue growth at score date": metrics.get("Revenue growth"),
        "Receivables growth at score date": metrics.get("Receivables growth"),
        "Receivables-revenue gap": metrics.get("Receivables minus revenue growth"),
        "OCF / net income": metrics.get("OCF / net income"),
        "Accruals proxy / average assets": metrics.get("Accruals proxy / average assets"),
        "Debt growth": metrics.get("Debt growth"),
        "Inventory growth": metrics.get("Inventory growth"),
        "Scoring version": SCORING_VERSION,
    }


def run_backtest(
    tickers_df: pd.DataFrame,
    years_per_company: int = 6,
    selected_sample: Optional[List[str]] = None,
    with_drawdown: bool = False,
    with_custom_extensions: bool = False,
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
            if with_custom_extensions and not filings.empty:
                # Extension normalization is opt-in because it requires several SEC
                # archive requests per filing. Point-in-time `filed` dates are preserved.
                extension_rows = filings.tail(int(max(1, years_per_company)) + 2)
                for _, extension_filing in extension_rows.iterrows():
                    try:
                        custom_observations, _ = extract_custom_xbrl_observations(company_row["cik"], extension_filing)
                        facts = merge_normalized_custom_facts(facts, custom_observations)
                    except Exception:
                        # Keep the standard-GAAP study usable; exclusions remain explicit in data coverage.
                        continue

            candidates = []
            filing_rows = list(filings.iterrows())
            for index, (_, filing) in enumerate(filing_rows):
                if _parse_timestamp(filing.get("reportDate")) is None:
                    continue
                next_filing = filing_rows[index + 1][1] if index + 1 < len(filing_rows) else None
                obs = historical_observation(ticker, filing, facts, next_filing=next_filing)
                if obs is not None:
                    obs["CIK"] = str(company_row["cik"])
                    candidates.append(obs)

            rows.extend(candidates[-int(max(1, years_per_company)):])
        except (SECDataError, KeyError, TypeError, ValueError):
            pass

        progress.progress(
            (i + 1) / len(selected),
            text=f"Processing {ticker} ({i + 1}/{len(selected)})…",
        )

    progress.empty()
    frame = pd.DataFrame(rows)
    if with_drawdown and not frame.empty:
        if yf is None:
            st.warning("Drawdown testing was requested, but yfinance is not installed. Install the dependencies in requirements_v7_2.txt and rerun.")
        else:
            with st.spinner("Fetching adjusted daily prices and calculating post-filing drawdowns…"):
                frame = attach_drawdowns(frame)
    return frame


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
    "EDGAR + XBRL → transparent rule-based financial profile → filing review → point-in-time research → outcome tests"
)
st.caption(f"Application version {APP_VERSION} · Scoring version {SCORING_VERSION}")
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
        min_value=1,
        max_value=10,
        value=6,
        help="Uses the most recent eligible historical observations per company. Set to 1 for one complete forward-outcome observation per company; higher values pool multiple company-years.",
    )
    st.caption(f"Scoring methodology: `{SCORING_VERSION}`")
    st.markdown("---")
    st.markdown("**Data source**")
    st.write("SEC EDGAR submissions + standard XBRL company facts")
    st.caption("Automated access should use a meaningful SEC User-Agent and comply with SEC fair-access guidance.")
    if "example.com" in SEC_USER_AGENT.lower():
        st.warning("Before repeated SEC use, set SEC_USER_AGENT in your environment or Streamlit secrets to an application name and a real contact email.")
    inspect_custom_tags = st.checkbox(
        "Inspect custom XBRL tags where standard concepts are missing",
        value=False,
        help="Downloads this filing's XBRL instance and extension label/presentation linkbases. High-confidence USD concepts may fill missing standard metrics; ambiguous matches are excluded. This adds SEC requests.",
    )

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
                custom_audit = pd.DataFrame()
                if inspect_custom_tags:
                    try:
                        custom_observations, custom_audit = extract_custom_xbrl_observations(cik, filing)
                        facts = merge_normalized_custom_facts(facts, custom_observations)
                    except Exception as custom_exc:
                        st.warning(f"Custom XBRL extension inspection was unavailable; standard US-GAAP scanning continued. ({type(custom_exc).__name__}: {custom_exc})")
                result = build_company_record(ticker, company_name, cik, facts, filing)
                result["custom_tag_audit"] = custom_audit.to_dict("records") if not custom_audit.empty else []
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
                    {"Metric": key, "Value": (f"{float(value):.1f} pp" if key == "Receivables minus revenue growth" and value is not None and pd.notna(value) else format_pct(value))}
                    for key, value in metrics.items()
                ]
            )
            st.dataframe(metric_frame, use_container_width=True, hide_index=True)
            if metrics.get("Accruals proxy / average assets") is not None:
                st.caption("Accruals proxy = (net income − operating cash flow) / average total assets. This is a screening proxy, not a complete accrual measure.")

        if result.get("custom_tag_audit"):
            st.subheader("Custom XBRL tag normalization audit")
            st.caption("Only high-confidence concepts with compatible USD units, period types and presentation roles are eligible to fill a missing standard metric. The table records excluded/ambiguous concepts too; it does not mean every candidate was used.")
            st.dataframe(pd.DataFrame(result["custom_tag_audit"]), use_container_width=True, hide_index=True)
            mapped_count = sum(1 for item in result["custom_tag_audit"] if item.get("Status") == "High-confidence candidate; eligible for fallback")
            st.caption(f"High-confidence candidate facts identified: {mapped_count}. A candidate only affects a metric when no usable standard US-GAAP fact exists for that metric and fiscal year.")

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

        if result.get("context_passages"):
            st.subheader("Filing context to review")
            st.caption("These are keyword-based research leads, not proof that a flag is benign or harmful. Verify the surrounding filing text.")
            for i, passage in enumerate(result["context_passages"], start=1):
                with st.expander(f"{i}. {passage['Topic']}"):
                    st.write(passage["Passage"])

        st.subheader("Subsequent-quarter collection evidence search")
        st.caption("Searches the first two 10-Q filings after this 10-K for receivables, collection, cash-receipt and payment-term language. A keyword match is not proof that most receivables were collected.")
        if st.button("Search next two 10-Qs", key="search_subsequent_10q"):
            try:
                quarter_submissions = get_submissions(result["cik"])
                quarter_passages = search_next_two_10q_collection_evidence(
                    result["cik"], quarter_submissions, result["filing_date"], max_filings=2
                )
                st.session_state["quarter_evidence"] = {"accession": result["accession"], "passages": quarter_passages}
            except Exception as exc:
                st.error(f"Subsequent-quarter evidence search failed: {type(exc).__name__}: {exc}")
        quarter_state = st.session_state.get("quarter_evidence")
        if quarter_state and quarter_state.get("accession") == result.get("accession"):
            if not quarter_state["passages"]:
                st.info("No matching passages were found in the first two available 10-Qs, or those filings were unavailable. This does not establish that collections did or did not occur.")
            else:
                st.dataframe(pd.DataFrame(quarter_state["passages"]), use_container_width=True, hide_index=True, column_config={
                    "SEC filing": st.column_config.LinkColumn("SEC filing", display_text="Open filing")
                })

        st.subheader("Disclosure-change analysis (TF-IDF / cosine similarity)")
        st.caption("Compares the current 10-K with the preceding standard 10-K. Text extraction is heuristic; inspect the excerpts before interpreting a change.")
        if st.button("Compare Risk Factors and MD&A with prior 10-K", key="compare_disclosures"):
            try:
                submissions_for_compare = get_submissions(result["cik"])
                filings_for_compare = annual_10ks(submissions_for_compare)
                filings_for_compare = filings_for_compare.sort_values("filingDate").reset_index(drop=True)
                current_matches = filings_for_compare.index[filings_for_compare["accessionNumber"].astype(str).eq(str(result["accession"]))].tolist()
                if not current_matches or current_matches[0] == 0:
                    st.warning("A preceding standard 10-K was not available in the returned filing history.")
                else:
                    current_row = filings_for_compare.iloc[current_matches[0]]
                    previous_row = filings_for_compare.iloc[current_matches[0] - 1]
                    previous_html, _ = get_filing_html(result["cik"], previous_row["accessionNumber"], previous_row["primaryDocument"])
                    current_html, _ = get_filing_html(result["cik"], current_row["accessionNumber"], current_row["primaryDocument"])
                    comparison = compare_disclosure_sections(extract_10k_sections(previous_html), extract_10k_sections(current_html))
                    st.session_state["disclosure_comparison"] = {
                        "accession": result["accession"],
                        "previous_date": str(previous_row["filingDate"].date()),
                        "current_date": str(current_row["filingDate"].date()),
                        "results": comparison,
                    }
            except Exception as exc:
                st.error(f"Disclosure comparison failed: {type(exc).__name__}: {exc}")
        disclosure_state = st.session_state.get("disclosure_comparison")
        if disclosure_state and disclosure_state.get("accession") == result.get("accession"):
            st.caption(f"Compared filing dates: {disclosure_state['previous_date']} → {disclosure_state['current_date']}")
            for section_name, section_result in disclosure_state["results"].items():
                st.markdown(f"**{section_name}**")
                if not section_result.get("available"):
                    st.warning(section_result.get("reason", "Section unavailable."))
                    continue
                metric_strip([
                    ("Cosine similarity", f"{section_result['cosine_similarity']:.3f}", "1.0 means identical TF-IDF direction; lower values indicate greater textual difference."),
                    ("Cosine distance", f"{section_result['cosine_distance']:.3f}", "One minus cosine similarity; not a measure of financial risk."),
                    ("Extracted characters", f"{section_result['current_chars']:,}", "Length of extracted current section."),
                ])
                st.dataframe(section_result["changes"].style.format({
                    "Change in TF-IDF weight": "{:+.4f}", "Prior weight": "{:.4f}", "Current weight": "{:.4f}"
                }), use_container_width=True, hide_index=True)
                with st.expander(f"Read passages around the largest {section_name} changes"):
                    st.dataframe(section_result["evidence_passages"], use_container_width=True, hide_index=True)

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
        ["20-company pilot", "100-company research sample", "10-company holdout", "Custom"],
        horizontal=True,
        key="backtest_sample_preset",
        help="Use the 20-company pilot for quick checks; use the 100-company sample for the final research run.",
    )
    if sample_preset == "20-company pilot":
        default_backtest = SAMPLE
    elif sample_preset == "100-company research sample":
        default_backtest = RESEARCH_SAMPLE_100
    elif sample_preset == "10-company holdout":
        default_backtest = HOLDOUT_SAMPLE_10
        st.caption("This is a newly specified, fixed holdout set disjoint from the original 20-company pilot; it does not verify a previously run ten-company experiment.")
    else:
        default_backtest = SAMPLE

    backtest_options = sorted(set(SAMPLE) | set(RESEARCH_SAMPLE_100) | set(HOLDOUT_SAMPLE_10) | set(tickers["ticker"].head(100).tolist()))
    backtest_sample = st.multiselect(
        "Backtest sample",
        options=backtest_options,
        default=default_backtest,
        key="backtest_sample",
    )

    include_drawdowns = st.checkbox(
        "Calculate post-filing 252-trading-day maximum drawdown (requires yfinance and internet access)",
        value=False,
        key="include_drawdowns",
    )
    include_custom_extensions = st.checkbox(
        "Inspect and use high-confidence custom XBRL tags as fallbacks (slower; more SEC requests)",
        value=False,
        key="include_custom_extensions",
    )
    if backtest_sample and st.button("Run historical backtest", type="primary", use_container_width=True):
        try:
            bt = run_backtest(tickers, years_per_company=backtest_years, selected_sample=backtest_sample, with_drawdown=include_drawdowns, with_custom_extensions=include_custom_extensions)
            st.session_state["backtest"] = bt
            track_event("backtest_run")
        except Exception as exc:
            st.error(f"Backtest failed: {type(exc).__name__}: {exc}")

    bt = st.session_state.get("backtest", pd.DataFrame()).copy()
    if not bt.empty:
        bt = bt.dropna(subset=["Score", "Forward revenue growth"]).reset_index(drop=True)
        st.metric("Historical observations", len(bt))
        st.caption("Each row is a company-year observation. Scores are calculated using information available by that historical filing date. Set the sidebar window to 1 for a single latest eligible observation per company.")

        # Cross-sectional view closest to the essay's 'twenty companies' framing:
        # one latest eligible observation per ticker with a known forward revenue outcome.
        latest_company_rows = bt.sort_values("Filing date").groupby("Ticker", as_index=False).tail(1).copy()
        flag_columns = ["Receivables flag", "Debt flag", "Cash-flow flag", "Inventory flag"]
        present_flag_columns = [col for col in flag_columns if col in latest_company_rows.columns]
        if present_flag_columns and not latest_company_rows.empty:
            latest_company_rows["Flag count"] = latest_company_rows[present_flag_columns].fillna(False).astype(bool).sum(axis=1)
            latest_company_rows["Any quantitative flag"] = latest_company_rows["Flag count"] > 0
            metric_strip([
                ("Companies with forward outcomes", str(latest_company_rows["Ticker"].nunique()), "One latest eligible company-year per ticker."),
                ("Companies with any flag", str(int(latest_company_rows["Any quantitative flag"].sum())), "At least one of the four quantitative flags in this snapshot."),
                ("Receivables flags", str(int(latest_company_rows["Receivables flag"].astype(bool).sum())), "Receivables growth gap exceeds 20 percentage points."),
            ])
            st.markdown("**Latest eligible observation per company**")
            company_view_cols = ["Ticker", "Filing date", "Flag count", *present_flag_columns,
                                 "Revenue growth at score date", "Receivables growth at score date",
                                 "Receivables-revenue gap", "Forward revenue growth"]
            if "Forward 252-day max drawdown (%)" in latest_company_rows.columns:
                company_view_cols.append("Forward 252-day max drawdown (%)")
            st.dataframe(latest_company_rows[company_view_cols].style.format({
                "Revenue growth at score date": "{:.1f}%",
                "Receivables growth at score date": "{:.1f}%",
                "Receivables-revenue gap": "{:.1f} pp",
                "Forward revenue growth": "{:.1f}%",
                "Forward 252-day max drawdown (%)": "{:.1f}%",
            }), use_container_width=True, hide_index=True)

        st.dataframe(
            bt.style.format({
                "Forward revenue growth": "{:.1f}%",
                "Revenue growth at score date": "{:.1f}%",
                "Receivables growth at score date": "{:.1f}%",
                "Receivables-revenue gap": "{:.1f} pp",
                "OCF / net income": "{:.0f}%",
                "Debt growth": "{:.1f}%",
                "Inventory growth": "{:.1f}%",
                "Accruals proxy / average assets": "{:.1f}%",
                "Forward 252-day max drawdown (%)": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        if "CIK" in bt.columns and "Accession number" in bt.columns:
            historical_case_labels = {
                f"{row['Ticker']} — FY{row['Fiscal year scored']} — filed {row['Filing date']} — {row['Accession number']}": idx
                for idx, row in bt.iterrows()
            }
            selected_historical_case = st.selectbox(
                "Historical filing to inspect for collection context",
                options=list(historical_case_labels.keys()),
                key="historical_collection_case",
                help="Choose a specific dated observation, such as the case where receivables growth and revenue growth diverged.",
            )
            if st.button("Search the following two 10-Qs for this observation", key="search_historical_10qs"):
                selected_row = bt.loc[historical_case_labels[selected_historical_case]]
                try:
                    case_submissions = get_submissions(str(selected_row["CIK"]))
                    case_passages = search_next_two_10q_collection_evidence(
                        str(selected_row["CIK"]), case_submissions, selected_row["Filing date"], max_filings=2
                    )
                    st.session_state["historical_quarter_evidence"] = {
                        "case": selected_historical_case,
                        "passages": case_passages,
                    }
                except Exception as exc:
                    st.error(f"Historical collection-context search failed: {type(exc).__name__}: {exc}")
            historical_quarter_state = st.session_state.get("historical_quarter_evidence")
            if historical_quarter_state and historical_quarter_state.get("case") == selected_historical_case:
                if historical_quarter_state["passages"]:
                    st.dataframe(pd.DataFrame(historical_quarter_state["passages"]), use_container_width=True, hide_index=True, column_config={
                        "SEC filing": st.column_config.LinkColumn("SEC filing", display_text="Open filing")
                    })
                else:
                    st.info("No matching passages were found in the first two subsequent 10-Qs, or those filings were unavailable. This does not establish that collections did or did not occur.")

        if len(bt) >= 5:
            if bt["Score"].nunique() >= 2:
                pearson = bt["Score"].corr(bt["Forward revenue growth"], method="pearson")
                spearman = bt["Score"].corr(bt["Forward revenue growth"], method="spearman")
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
            else:
                st.info("The overall score has no variation in this sample, so its correlation is undefined. Individual-flag tests below are still reported.")

            st.subheader("Look-ahead-bias sensitivity audit")
            st.caption("The unrestricted score uses facts that may have been filed later or restated; it is an invalid diagnostic, not a valid backtest. Both correlations use the same observation rows and forward revenue-growth outcome.")
            if "Naive score (invalid look-ahead diagnostic)" in bt.columns:
                valid_compare = bt.dropna(subset=["Forward revenue growth", "Score", "Naive score (invalid look-ahead diagnostic)"])
                audit_rows = []
                for label, column in [
                    ("Filing-date-constrained score", "Score"),
                    ("Naive score (invalid diagnostic)", "Naive score (invalid look-ahead diagnostic)"),
                ]:
                    pearson_a = valid_compare[column].corr(valid_compare["Forward revenue growth"], method="pearson") if len(valid_compare) >= 3 else np.nan
                    spearman_a = valid_compare[column].corr(valid_compare["Forward revenue growth"], method="spearman") if len(valid_compare) >= 3 else np.nan
                    audit_rows.append({"Score version": label, "Observations": len(valid_compare), "Pearson r": pearson_a, "Spearman ρ": spearman_a})
                st.dataframe(pd.DataFrame(audit_rows).style.format({"Pearson r": "{:.3f}", "Spearman ρ": "{:.3f}"}), use_container_width=True, hide_index=True)
                st.caption("A weaker date-aware correlation is not guaranteed. The audit reports what the data show; it does not tune the result to match an essay narrative.")

            st.subheader("Flag-level forward revenue-growth tests")
            flag_growth = summarize_flag_outcomes(bt, "Forward revenue growth")
            if flag_growth.empty:
                st.info("There are not enough valid observations to compare individual flags yet.")
            else:
                st.dataframe(flag_growth.style.format({
                    "Flagged median": "{:.2f}%", "Unflagged median": "{:.2f}%",
                    "Median difference (flagged − unflagged)": "{:+.2f}%",
                    "Flagged mean": "{:.2f}%", "Unflagged mean": "{:.2f}%",
                    "Mean difference (flagged − unflagged)": "{:+.2f}%",
                    "Pearson r (binary flag)": "{:.3f}", "Spearman ρ (binary flag)": "{:.3f}",
                    "Median-difference 95% CI low (cluster bootstrap)": "{:.2f}%",
                    "Median-difference 95% CI high (cluster bootstrap)": "{:.2f}%"
                }), use_container_width=True, hide_index=True)

            # Per-signal incidence across the historical sample.
            st.subheader("Signal incidence")
            signal_rows = []
            for label, threshold in [
                ("Receivables growth exceeded revenue growth by >20pp", "Receivables-revenue gap"),
                ("Debt growth ≥20%", "Debt growth"),
                ("Inventory growth ≥25%", "Inventory growth"),
                ("OCF / net income <60%", "OCF / net income"),
            ]:
                series = bt[threshold]
                if threshold == "Receivables-revenue gap":
                    triggered = series.gt(20).fillna(False)
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

            if "Forward 252-day max drawdown (%)" in bt.columns:
                st.subheader("Post-filing stock drawdown outcomes")
                complete_drawdowns = bt["Forward 252-day max drawdown (%)"].notna().sum()
                st.metric("Complete 252-trading-day price windows", f"{complete_drawdowns} / {len(bt)}")
                if complete_drawdowns:
                    st.dataframe(summarize_flag_outcomes(bt, "Forward 252-day max drawdown (%)").style.format({
                        "Flagged median": "{:.2f}%", "Unflagged median": "{:.2f}%",
                        "Median difference (flagged − unflagged)": "{:+.2f}%",
                        "Flagged mean": "{:.2f}%", "Unflagged mean": "{:.2f}%",
                        "Mean difference (flagged − unflagged)": "{:+.2f}%",
                        "Pearson r (binary flag)": "{:.3f}", "Spearman ρ (binary flag)": "{:.3f}",
                        "Median-difference 95% CI low (cluster bootstrap)": "{:.2f}%",
                        "Median-difference 95% CI high (cluster bootstrap)": "{:.2f}%"
                    }), use_container_width=True, hide_index=True)
                    tail = drawdown_tail_summary(bt)
                    if not tail.empty:
                        st.markdown("**Flag prevalence in the worst drawdown quartile**")
                        st.dataframe(tail.style.format({
                            "Flag rate in worst quartile (%)": "{:.1f}%", "Flag rate elsewhere (%)": "{:.1f}%",
                            "Difference (percentage points)": "{:+.1f}", "Worst-quartile cutoff (%)": "{:.1f}%"
                        }), use_container_width=True, hide_index=True)
                st.caption("Drawdown is a stock-price path statistic, not evidence that a filing flag caused a decline. Recent or incomplete one-year windows remain missing.")

        else:
            st.info("Not enough observations for meaningful association tests. Expand the sample or historical window.")

        safe_download_df(bt, "sec_risk_lab_historical_backtest.csv", "Download historical dataset")


# ============================================================
# METHODOLOGY
# ============================================================

with tab4:
    st.subheader("Methodology & Research Notes")
    st.markdown(
        f"**Scoring version:** `{SCORING_VERSION}`\n\n"
        "The quantitative score starts at 100 and subtracts points for four predefined accounting patterns. "
        "Version 7.2 uses a receivables-growth gap greater than 20 percentage points, matching the stated research rule. "
        "The score is a screening framework, not a calibrated default probability or investment recommendation."
    )

    st.markdown("### Quantitative scoring")
    scoring_table = pd.DataFrame([
        ["Receivables growth > revenue growth + 20pp", 12, "Working-capital quality"],
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
        "Optional custom-tag fallback inspects the filing's XBRL instance, labels and presentation roles; it fills a missing metric only for high-confidence USD/unit/period/context matches, and otherwise excludes the candidate.",
        "A simple accruals proxy is reported only when net income, operating cash flow, and positive current/prior total assets are available.",
        "The new 10-company holdout is fixed and separate from the original 20-company pilot; it does not stand in for an older unpublished test sample.",
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
        "The primary backtest compares a filing-date-constrained score from fiscal year t with reported revenue growth from t to t+1, "
        "then reports individual-flag outcomes separately. Optional price research measures maximum drawdown over the next 252 trading-day closes "
        "after each 10-K filing date. Price windows without a full 252 closes are excluded from outcome comparisons. These analyses test associations; "
        "they do not establish causation, fraud prediction, or an investable trading strategy. The look-ahead diagnostic intentionally uses unrestricted facts "
        "and is displayed only to audit sensitivity, never as valid predictive evidence."
    )
    st.markdown("### Accrual proxy")
    st.write(
        "The accruals proxy is (net income − operating cash flow) / average total assets × 100. It is a simple, transparent proxy, "
        "not a complete accrual accounting decomposition and is not currently assigned a separate scoring penalty."
    )
    st.markdown("### Disclosure text comparison")
    st.write(
        "The optional disclosure comparison extracts Risk Factors and MD&A heuristically from consecutive 10-Ks, then compares TF-IDF representations "
        "with cosine similarity. It surfaces terms with the largest changes in TF-IDF weight for manual review. Section extraction can fail on unusual filing layouts; "
        "similarity is a text statistic, not a measure of risk or management intent."
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
