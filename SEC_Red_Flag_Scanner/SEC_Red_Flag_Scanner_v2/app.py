import re
import pandas as pd
import requests
import streamlit as st

SEC_HEADERS = {
    "User-Agent": "SEC Filing Red-Flag Scanner/1.0 research-demo contact@example.com"
}

# ---------- SEC ----------
@st.cache_data(ttl=3600)
def get_tickers():
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS, timeout=20
    )
    r.raise_for_status()
    return pd.DataFrame([
        {"ticker": v["ticker"], "name": v["title"],
         "cik": str(v["cik_str"]).zfill(10)}
        for v in r.json().values()
    ])

@st.cache_data(ttl=3600)
def get_submissions(cik):
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS, timeout=20
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def get_companyfacts(cik):
    r = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=SEC_HEADERS, timeout=30
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def get_filing(cik, accession, document):
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{document}"
    )
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
    return text, url

# ---------- XBRL financial data ----------
TAG_CANDIDATES = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "operating_cash_flow": [
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
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
    ],
    "debt": [
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    ],
    "assets": [
        "Assets",
    ],
    "liabilities": [
        "Liabilities",
    ],
}

def extract_series(facts, metric):
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in TAG_CANDIDATES[metric]:
        if tag not in usgaap:
            continue
        units = usgaap[tag].get("units", {})
        unit = "USD" if "USD" in units else next(iter(units), None)
        if not unit:
            continue
        rows = units[unit]
        cleaned = []
        for x in rows:
            if x.get("form") not in ("10-K", "10-Q"):
                continue
            if "fy" not in x:
                continue
            try:
                val = float(x["val"])
            except Exception:
                continue
            cleaned.append({
                "fy": int(x["fy"]),
                "fp": x.get("fp"),
                "form": x.get("form"),
                "end": x.get("end"),
                "val": val,
            })
        if cleaned:
            df = pd.DataFrame(cleaned).drop_duplicates(
                subset=["fy", "form", "end"], keep="last"
            )
            return df.sort_values(["fy", "end"])
    return pd.DataFrame(columns=["fy", "form", "end", "val"])

def latest_annual(facts, metric, n=3):
    df = extract_series(facts, metric)
    if df.empty:
        return df
    df = df[df["form"] == "10-K"].copy()
    return df.drop_duplicates("fy", keep="last").tail(n)

def pct_growth(old, new):
    if old is None or old == 0 or new is None:
        return None
    return (new - old) / abs(old) * 100

# ---------- Transparent risk framework ----------
def filing_signals(text):
    t = text.lower()
    rules = [
        ("Accounting / controls", "Material weakness", ["material weakness", "restatement"], 3),
        ("Accounting / controls", "Going-concern language", ["going concern", "substantial doubt"], 3),
        ("Governance", "Related-party activity", ["related party", "related-party"], 2),
        ("Governance", "Customer concentration", ["customer concentration", "concentration of customers"], 2),
        ("Liquidity", "Debt covenant language", ["debt covenant", "covenant violation"], 2),
        ("Liquidity", "Liquidity constraints", ["liquidity constraints", "liquidity risk"], 2),
        ("Business quality", "Restructuring activity", ["restructuring charge", "restructuring"], 1),
        ("Business quality", "Asset impairment", ["impairment charge", "impairment"], 1),
        ("Outlook", "Negative guidance language", ["lowered our guidance", "below our previous guidance", "headwinds"], 1),
    ]
    hits = []
    evidence = []
    for category, label, terms, severity in rules:
        hit = next((term for term in terms if term in t), None)
        if hit:
            hits.append({
                "Category": category,
                "Signal": label,
                "Severity": severity,
                "Trigger": hit
            })
            i = t.find(hit)
            evidence.append((label, text[max(0, i-260):i+520]))
    return hits, evidence[:8]

def quantitative_signals(facts):
    signals = []
    metrics = {}
    for metric in TAG_CANDIDATES:
        df = latest_annual(facts, metric, 3)
        if not df.empty:
            metrics[metric] = df

    def last_two(metric):
        df = metrics.get(metric)
        if df is None or len(df) < 2:
            return None, None
        return float(df.iloc[-2]["val"]), float(df.iloc[-1]["val"])

    # Revenue vs receivables growth
    rev0, rev1 = last_two("revenue")
    rec0, rec1 = last_two("receivables")
    if rev0 is not None and rec0 is not None:
        rg = pct_growth(rev0, rev1)
        qg = pct_growth(rec0, rec1)
        if rg is not None and qg is not None:
            metrics["_revenue_growth"] = rg
            metrics["_receivables_growth"] = qg
            if qg - rg >= 15:
                signals.append(("Business quality", "Receivables growing materially faster than revenue", 2,
                                f"Revenue growth {rg:.1f}% vs. receivables growth {qg:.1f}%"))

    # Net income vs operating cash flow
    ni0, ni1 = last_two("net_income")
    ocf0, ocf1 = last_two("operating_cash_flow")
    if ni1 is not None and ocf1 is not None:
        metrics["_latest_net_income"] = ni1
        metrics["_latest_ocf"] = ocf1
        if ni1 > 0 and ocf1 < ni1 * 0.60:
            signals.append(("Cash flow", "Operating cash flow trails net income", 2,
                            "Latest OCF is below 60% of net income"))

    # Debt growth
    d0, d1 = last_two("debt")
    if d0 is not None and d1 is not None:
        dg = pct_growth(d0, d1)
        metrics["_debt_growth"] = dg
        if dg is not None and dg >= 20:
            signals.append(("Liquidity", "Debt increased materially year-over-year", 2,
                            f"Debt growth {dg:.1f}%"))

    # Inventory growth
    inv0, inv1 = last_two("inventory")
    if inv0 is not None and inv1 is not None:
        ig = pct_growth(inv0, inv1)
        metrics["_inventory_growth"] = ig
        if ig is not None and ig >= 25:
            signals.append(("Business quality", "Inventory increased materially", 1,
                            f"Inventory growth {ig:.1f}%"))

    return signals, metrics

def make_score(filing_hits, quant_hits):
    # Start at 100; cap deductions so the score remains interpretable.
    total = sum(x["Severity"] for x in filing_hits) + sum(x[2] for x in quant_hits)
    score = max(0, min(100, 100 - total * 7))
    level = "Low" if score >= 80 else "Moderate" if score >= 60 else "Elevated"
    return score, level

def build_report(company, filing_date, score, level, filing_hits, quant_hits, evidence):
    out = [
        f"# SEC Filing Red-Flag Report — {company}",
        f"**Filing date:** {filing_date}",
        f"**Financial risk score:** {score}/100 ({level})",
        "",
        "## Methodology",
        "Score combines transparent filing-language signals with basic SEC XBRL financial-statement tests. It is a screening tool, not an investment recommendation.",
        "",
        "## Filing signals"
    ]
    out += [
        f"- **{x['Category']} — {x['Signal']}** (severity {x['Severity']}; trigger `{x['Trigger']}`)"
        for x in filing_hits
    ] or ["No major filing-language signals detected."]
    out += ["", "## Quantitative signals"]
    out += [
        f"- **{cat} — {label}** (severity {sev}): {detail}"
        for cat, label, sev, detail in quant_hits
    ] or ["No quantitative thresholds were triggered."]
    out += ["", "## Evidence"]
    out += [f"> **{label}:** {snippet}" for label, snippet in evidence] or ["No evidence snippets captured."]
    out += [
        "", "## Analyst follow-ups",
        "- Reconcile reported earnings with operating and free cash flow.",
        "- Inspect debt maturities, covenants and refinancing needs.",
        "- Investigate working-capital movements relative to revenue.",
        "- Read the relevant footnotes and risk factors before forming a view.",
        "",
        "*Educational research prototype; false positives and false negatives are possible.*"
    ]
    return "\n".join(out)

# ---------- App ----------
st.set_page_config(page_title="SEC Filing Red-Flag Scanner", page_icon="🔎", layout="wide")
st.title("🔎 SEC Filing Red-Flag Scanner")
st.caption("Automated due-diligence prototype: SEC EDGAR → filing language + XBRL financial data → risk framework → analyst questions.")

try:
    tickers = get_tickers()
except Exception as e:
    st.error(f"SEC company list unavailable: {e}")
    st.stop()

q = st.text_input("Search company", placeholder="Apple, NVIDIA, Tesla...")
matches = tickers if not q else tickers[
    tickers["ticker"].str.contains(q.upper(), na=False) |
    tickers["name"].str.contains(q, case=False, na=False)
]
matches = matches.head(25)
options = {f"{r.ticker} — {r.name}": (r.ticker, r.cik, r.name) for _, r in matches.iterrows()}
choice = st.selectbox("Company", list(options) if options else ["No matches"])

if st.button("Run red-flag scan", type="primary"):
    ticker, cik, company = options[choice]
    try:
        subs = get_submissions(cik)
        recent = pd.DataFrame(subs["filings"]["recent"])
        annual = recent[recent["form"] == "10-K"].sort_values("filingDate", ascending=False)
        row = annual.iloc[0]

        with st.spinner("Reading SEC filing and XBRL financial data..."):
            text, url = get_filing(cik, row.accessionNumber, row.primaryDocument)
            facts = get_companyfacts(cik)
            filing_hits, evidence = filing_signals(text)
            quant_hits, metrics = quantitative_signals(facts)
            score, level = make_score(filing_hits, quant_hits)

        st.session_state.update(
            score=score, level=level, filing_hits=filing_hits,
            quant_hits=quant_hits, evidence=evidence, url=url,
            report=build_report(company, row.filingDate, score, level, filing_hits, quant_hits, evidence),
            company=company, filing_date=str(row.filingDate)
        )
    except Exception as e:
        st.error(f"SEC analysis failed: {e}")
        st.stop()

if "score" in st.session_state:
    a, b, c = st.columns(3)
    a.metric("Financial risk score", f"{st.session_state['score']}/100")
    b.metric("Risk level", st.session_state["level"])
    b.caption("Higher = stronger financial profile")
    c.metric("Signals", len(st.session_state["filing_hits"]) + len(st.session_state["quant_hits"]))

    st.subheader("Risk signals")
    rows = []
    for x in st.session_state["filing_hits"]:
        rows.append({
            "Category": x["Category"], "Signal": x["Signal"],
            "Severity": x["Severity"], "Evidence": x["Trigger"]
        })
    for cat, label, sev, detail in st.session_state["quant_hits"]:
        rows.append({
            "Category": cat, "Signal": label,
            "Severity": sev, "Evidence": detail
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("No major heuristic signals detected.")

    st.subheader("Evidence from the filing")
    for i, (label, snippet) in enumerate(st.session_state["evidence"], 1):
        with st.expander(f"{i}. {label}"):
            st.write(snippet)

    st.subheader("Analyst follow-ups")
    for item in [
        "Reconcile reported earnings with operating/free cash flow.",
        "Inspect debt maturities, covenants and refinancing needs.",
        "Compare receivables and inventory growth with revenue growth.",
        "Read the relevant footnotes and risk factors."
    ]:
        st.write("• " + item)

    st.download_button(
        "Download analyst report",
        st.session_state["report"],
        file_name=f"{str(st.session_state['company']).replace(' ','_')}_red_flags.md",
        mime="text/markdown"
    )
    st.markdown(f"[Open SEC filing]({st.session_state['url']})")
