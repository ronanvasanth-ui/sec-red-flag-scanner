import re
import time
import pandas as pd
import requests
import streamlit as st

# ============================================================
# SEC RED-FLAG SCANNER v3
# ============================================================

SEC_HEADERS = {
    "User-Agent": "SEC Filing Red-Flag Scanner/1.0 student-research-demo contact@example.com"
}

# Replace contact@example.com above with your own email before public use.

# ---------- SEC data ----------
@st.cache_data(ttl=3600)
def get_tickers():
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    return pd.DataFrame([
        {
            "ticker": v["ticker"],
            "name": v["title"],
            "cik": str(v["cik_str"]).zfill(10),
        }
        for v in r.json().values()
    ])

@st.cache_data(ttl=3600)
def get_submissions(cik):
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def get_companyfacts(cik):
    r = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def get_filing(cik, accession, document):
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{document}"
    )
    r = requests.get(url, headers=SEC_HEADERS, timeout=40)
    r.raise_for_status()
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
    return text, url

# ---------- XBRL ----------
TAG_CANDIDATES = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
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
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
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

        rows = []
        for x in units[unit]:
            if x.get("form") != "10-K" or "fy" not in x:
                continue
            try:
                value = float(x["val"])
            except Exception:
                continue
            rows.append({
                "fy": int(x["fy"]),
                "end": x.get("end"),
                "val": value,
            })

        if rows:
            df = pd.DataFrame(rows)
            return df.sort_values(["fy", "end"]).drop_duplicates("fy", keep="last")

    return pd.DataFrame(columns=["fy", "end", "val"])

def latest_two(facts, metric):
    df = extract_series(facts, metric)
    if len(df) < 2:
        return None
    return df.iloc[-2], df.iloc[-1]

def growth(old, new):
    if old is None or old == 0 or new is None:
        return None
    return (new - old) / abs(old) * 100

# ---------- Filing-language review signals ----------
# These are deliberately labeled "review signals" rather than automatically
# treated as proof of financial distress.
FILING_RULES = [
    ("Accounting / controls", "Material weakness language",
     ["material weakness"], 3),
    ("Accounting / controls", "Restatement language",
     ["restatement", "restated financial statements"], 3),
    ("Liquidity", "Going-concern language",
     ["going concern", "substantial doubt about"], 4),
    ("Liquidity", "Debt covenant language",
     ["debt covenant", "covenant violation"], 2),
    ("Governance", "Related-party language",
     ["related party", "related-party"], 1),
    ("Business quality", "Customer concentration language",
     ["customer concentration", "concentration of customers"], 1),
    ("Business quality", "Impairment language",
     ["impairment charge"], 1),
    ("Business quality", "Restructuring language",
     ["restructuring charge"], 1),
    ("Outlook", "Negative guidance language",
     ["lowered our guidance", "below our previous guidance"], 1),
]

def filing_review_signals(text):
    t = text.lower()
    hits, evidence = [], []

    for category, label, terms, severity in FILING_RULES:
        term = next((x for x in terms if x in t), None)
        if term:
            hits.append({
                "Category": category,
                "Signal": label,
                "Severity": severity,
                "Trigger": term,
            })
            i = t.find(term)
            snippet = text[max(0, i - 260): i + 520]
            evidence.append((label, snippet))

    return hits, evidence[:8]

# ---------- Quantitative risk framework ----------
def quantitative_analysis(facts):
    risks = []
    metrics = {}

    pairs = {
        "revenue": latest_two(facts, "revenue"),
        "receivables": latest_two(facts, "receivables"),
        "net_income": latest_two(facts, "net_income"),
        "operating_cash_flow": latest_two(facts, "operating_cash_flow"),
        "debt": latest_two(facts, "debt"),
        "inventory": latest_two(facts, "inventory"),
    }

    def pair_values(name):
        pair = pairs.get(name)
        if not pair:
            return None, None
        return float(pair[0]["val"]), float(pair[1]["val"])

    # 1. Receivables growth vs revenue growth
    rev0, rev1 = pair_values("revenue")
    rec0, rec1 = pair_values("receivables")
    if rev0 is not None and rec0 is not None:
        rev_growth = growth(rev0, rev1)
        rec_growth = growth(rec0, rec1)
        if rev_growth is not None and rec_growth is not None:
            metrics["Revenue growth"] = rev_growth
            metrics["Receivables growth"] = rec_growth

            gap = rec_growth - rev_growth
            if gap >= 15:
                risks.append({
                    "Category": "Working capital",
                    "Signal": "Receivables growing materially faster than revenue",
                    "Points": 12,
                    "Evidence": f"Revenue growth {rev_growth:.1f}% vs. receivables growth {rec_growth:.1f}%",
                })

    # 2. Operating cash flow vs net income
    ni0, ni1 = pair_values("net_income")
    ocf0, ocf1 = pair_values("operating_cash_flow")
    if ni1 is not None and ocf1 is not None:
        metrics["Latest net income"] = ni1
        metrics["Latest operating cash flow"] = ocf1

        if ni1 > 0 and ocf1 < ni1 * 0.60:
            ratio = ocf1 / ni1 * 100
            risks.append({
                "Category": "Cash flow",
                "Signal": "Operating cash flow materially trails net income",
                "Points": 12,
                "Evidence": f"Operating cash flow is {ratio:.0f}% of net income",
            })

    # 3. Debt growth
    d0, d1 = pair_values("debt")
    if d0 is not None and d1 is not None:
        debt_growth = growth(d0, d1)
        metrics["Debt growth"] = debt_growth

        if debt_growth is not None and debt_growth >= 20:
            risks.append({
                "Category": "Liquidity",
                "Signal": "Debt increased materially year-over-year",
                "Points": 8,
                "Evidence": f"Debt growth {debt_growth:.1f}%",
            })

    # 4. Inventory growth
    inv0, inv1 = pair_values("inventory")
    if inv0 is not None and inv1 is not None:
        inventory_growth = growth(inv0, inv1)
        metrics["Inventory growth"] = inventory_growth

        if inventory_growth is not None and inventory_growth >= 25:
            risks.append({
                "Category": "Business quality",
                "Signal": "Inventory increased materially",
                "Points": 6,
                "Evidence": f"Inventory growth {inventory_growth:.1f}%",
            })

    # Higher score = stronger financial profile.
    risk_points = sum(x["Points"] for x in risks)
    score = max(0, 100 - min(60, risk_points))

    if score >= 85:
        level = "Low"
    elif score >= 70:
        level = "Moderate"
    elif score >= 50:
        level = "Elevated"
    else:
        level = "High"

    return score, level, risks, metrics

def analyze_company(cik, company, filing_row):
    text, filing_url = get_filing(
        cik,
        filing_row["accessionNumber"],
        filing_row["primaryDocument"],
    )
    facts = get_companyfacts(cik)

    review_signals, evidence = filing_review_signals(text)
    score, level, quant_risks, metrics = quantitative_analysis(facts)

    return {
        "company": company,
        "score": score,
        "level": level,
        "quant_risks": quant_risks,
        "review_signals": review_signals,
        "evidence": evidence,
        "metrics": metrics,
        "filing_date": str(filing_row["filingDate"]),
        "filing_url": filing_url,
    }

# ---------- Research sample ----------
RESEARCH_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "MA",
    "WMT", "COST", "KO", "NFLX", "DIS",
    "ORCL", "PEP", "ADBE", "CRM", "INTC",
]

def run_research(tickers_df):
    rows = []
    progress = st.progress(0, text="Starting cross-company study...")

    for i, ticker in enumerate(RESEARCH_TICKERS):
        match = tickers_df[tickers_df["ticker"] == ticker]
        if match.empty:
            continue

        r = match.iloc[0]
        try:
            subs = get_submissions(r["cik"])
            recent = pd.DataFrame(subs["filings"]["recent"])
            annual = recent[recent["form"] == "10-K"].sort_values(
                "filingDate", ascending=False
            )
            if annual.empty:
                continue

            result = analyze_company(r["cik"], r["name"], annual.iloc[0])

            rows.append({
                "Ticker": ticker,
                "Company": r["name"],
                "Risk score": result["score"],
                "Risk level": result["level"],
                "Quantitative flags": len(result["quant_risks"]),
                "Filing review signals": len(result["review_signals"]),
                "Primary categories": ", ".join(
                    sorted(set(x["Category"] for x in result["quant_risks"]))
                ) or "None",
            })
        except Exception:
            # Keep the study running if one issuer has an unusual filing structure.
            pass

        progress.progress(
            (i + 1) / len(RESEARCH_TICKERS),
            text=f"Analyzing {ticker} ({i+1}/{len(RESEARCH_TICKERS)})..."
        )
        time.sleep(0.15)

    progress.empty()
    return pd.DataFrame(rows)

def build_report(result):
    lines = [
        f"# SEC Filing Red-Flag Report — {result['company']}",
        f"**Filing date:** {result['filing_date']}",
        f"**Financial profile score:** {result['score']}/100 ({result['level']})",
        "",
        "## What the score means",
        "The score starts at 100 and deducts points only for quantitative financial tests that cross predefined thresholds. Filing-language matches are shown separately as review signals because a keyword alone does not prove financial distress.",
        "",
        "## Quantitative risk signals",
    ]

    if result["quant_risks"]:
        lines += [
            f"- **{x['Category']} — {x['Signal']}** (-{x['Points']} points): {x['Evidence']}"
            for x in result["quant_risks"]
        ]
    else:
        lines.append("- No quantitative thresholds were triggered.")

    lines += ["", "## Filing-language review signals"]

    if result["review_signals"]:
        lines += [
            f"- **{x['Category']} — {x['Signal']}** — trigger: `{x['Trigger']}`"
            for x in result["review_signals"]
        ]
    else:
        lines.append("- No predefined review signals detected.")

    lines += ["", "## Evidence"]

    if result["evidence"]:
        lines += [
            f"> **{label}:** {snippet}"
            for label, snippet in result["evidence"]
        ]
    else:
        lines.append("No evidence snippets captured.")

    lines += [
        "",
        "## Analyst follow-ups",
        "- Reconcile reported earnings with operating and free cash flow.",
        "- Investigate working-capital movements relative to revenue.",
        "- Inspect debt maturities, covenants and refinancing needs.",
        "- Read the relevant footnotes and risk factors in context.",
        "",
        "*Educational research prototype. False positives and false negatives are possible; this is not investment advice.*",
    ]

    return "\n".join(lines)

# ============================================================
# UI
# ============================================================
st.set_page_config(
    page_title="SEC Filing Red-Flag Scanner",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 SEC Filing Red-Flag Scanner")
st.caption(
    "SEC EDGAR → XBRL financial data + filing-language review → transparent risk framework → analyst questions"
)

tab1, tab2 = st.tabs(["🔎 Company Scanner", "📊 Cross-Company Research"])

# ---------- Company scanner ----------
with tab1:
    try:
        tickers = get_tickers()
    except Exception as e:
        st.error(f"SEC company list unavailable: {e}")
        st.stop()

    query = st.text_input(
        "Search company",
        placeholder="Apple, NVIDIA, Tesla...",
        key="company_search",
    )

    matches = tickers
    if query:
        matches = tickers[
            tickers["ticker"].str.contains(query.upper(), na=False)
            | tickers["name"].str.contains(query, case=False, na=False)
        ]

    matches = matches.head(25)
    options = {
        f"{r.ticker} — {r.name}": (r.ticker, r.cik, r.name)
        for _, r in matches.iterrows()
    }

    choice = st.selectbox(
        "Company",
        list(options) if options else ["No matches"],
        key="company_choice",
    )

    if st.button("Run red-flag scan", type="primary"):
        ticker, cik, company = options[choice]

        try:
            with st.spinner("Reading SEC filing and XBRL financial data..."):
                submissions = get_submissions(cik)
                recent = pd.DataFrame(submissions["filings"]["recent"])
                annual = recent[
                    recent["form"] == "10-K"
                ].sort_values("filingDate", ascending=False)

                if annual.empty:
                    st.error("No recent 10-K found.")
                    st.stop()

                result = analyze_company(cik, company, annual.iloc[0])
                result["report"] = build_report(result)
                st.session_state["company_result"] = result

        except Exception as e:
            st.error(f"SEC analysis failed: {e}")

    if "company_result" in st.session_state:
        result = st.session_state["company_result"]

        a, b, c = st.columns(3)
        a.metric("Financial profile score", f"{result['score']}/100")
        b.metric("Risk level", result["level"])
        c.metric(
            "Quantitative flags",
            len(result["quant_risks"])
        )

        st.caption(
            "Higher score = stronger profile. Filing-language matches are review signals, not automatic proof of financial risk."
        )

        st.subheader("Quantitative risk signals")

        if result["quant_risks"]:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Category": x["Category"],
                        "Signal": x["Signal"],
                        "Points": x["Points"],
                        "Evidence": x["Evidence"],
                    }
                    for x in result["quant_risks"]
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No quantitative thresholds were triggered.")

        st.subheader("Filing-language review signals")

        if result["review_signals"]:
            st.dataframe(
                pd.DataFrame(result["review_signals"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No predefined review signals detected.")

        st.subheader("Evidence from the filing")

        for i, (label, snippet) in enumerate(result["evidence"], 1):
            with st.expander(f"{i}. {label}"):
                st.write(snippet)

        st.subheader("Analyst follow-ups")

        for item in [
            "Reconcile reported earnings with operating/free cash flow.",
            "Investigate working-capital movements relative to revenue.",
            "Inspect debt maturities, covenants and refinancing needs.",
            "Read the relevant footnotes and risk factors in context.",
        ]:
            st.write("• " + item)

        st.download_button(
            "Download analyst report",
            result["report"],
            file_name="SEC_red_flag_report.md",
            mime="text/markdown",
        )

        st.markdown(f"[Open SEC filing]({result['filing_url']})")

# ---------- Research mode ----------
with tab2:
    st.subheader("📊 Cross-Company Research Study")

    st.write(
        """
        **Research question:** Can a transparent, automated financial-risk framework
        identify meaningful differences in financial profiles across major public companies?

        This study applies exactly the same quantitative thresholds to every company,
        creating a reproducible cross-company dataset.
        """
    )

    st.info(
        "The research sample is intentionally fixed at 20 large public companies so "
        "the methodology is reproducible rather than cherry-picked."
    )

    if st.button("Run 20-company study", type="primary"):
        try:
            tickers = get_tickers()
            study = run_research(tickers)

            if study.empty:
                st.error("The study returned no results.")
            else:
                st.session_state["study"] = study
        except Exception as e:
            st.error(f"Research study failed: {e}")

    if "study" in st.session_state:
        study = st.session_state["study"].copy()

        avg_score = study["Risk score"].mean()
        high_risk = (study["Risk score"] < 70).sum()
        avg_quant = study["Quantitative flags"].mean()

        a, b, c = st.columns(3)
        a.metric("Companies analyzed", len(study))
        b.metric("Average score", f"{avg_score:.1f}/100")
        c.metric("Elevated/high-risk profiles", int(high_risk))

        st.subheader("Results")

        ranked = study.sort_values("Risk score")
        st.dataframe(
            ranked,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Risk score distribution")

        chart_df = ranked.set_index("Ticker")[["Risk score"]]
        st.bar_chart(chart_df)

        st.subheader("Quantitative flags by company")

        flags_df = ranked.set_index("Ticker")[[
            "Quantitative flags",
            "Filing review signals",
        ]]
        st.bar_chart(flags_df)

        st.subheader("What the study can tell us")

        st.write(
            f"The sample produced an average quantitative profile score of "
            f"**{avg_score:.1f}/100**. "
            f"**{int(high_risk)} of {len(study)} companies** fell below the "
            f"70-point threshold. The framework generated an average of "
            f"**{avg_quant:.1f} quantitative flags per company**."
        )

        st.warning(
            "This is descriptive research, not proof that the framework predicts "
            "future stock returns or financial distress. A stronger next stage is "
            "to compare these scores with subsequent financial outcomes."
        )

        csv = study.to_csv(index=False)

        st.download_button(
            "Download research dataset (CSV)",
            csv,
            file_name="sec_red_flag_cross_company_study.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Methodology")

        st.markdown(
            """
**Quantitative thresholds**

- Receivables growth ≥ 15 percentage points above revenue growth: **12 points**
- Operating cash flow < 60% of net income: **12 points**
- Debt growth ≥ 20%: **8 points**
- Inventory growth ≥ 25%: **6 points**

**Score interpretation**

- 85–100: Low
- 70–84: Moderate
- 50–69: Elevated
- Below 50: High

Filing-language matches are intentionally separated from the quantitative score.
The reason is methodological: the mere appearance of words such as “impairment,”
“covenant,” or “material weakness” does not establish that the company currently
has a financial problem. Those matches therefore become **analyst review signals**
rather than automatic proof of risk.
"""
        )
