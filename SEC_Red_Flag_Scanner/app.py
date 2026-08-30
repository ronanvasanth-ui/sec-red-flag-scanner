import re
import time
import pandas as pd
import requests
import streamlit as st

SEC_HEADERS = {
    "User-Agent": "SEC Filing Red-Flag Scanner/1.0 student-research-demo contact@example.com"
}

# ============================================================
# SEC DATA
# ============================================================

@st.cache_data(ttl=3600)
def get_tickers():
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=SEC_HEADERS, timeout=20)
    r.raise_for_status()
    return pd.DataFrame([
        {"ticker": v["ticker"], "name": v["title"],
         "cik": str(v["cik_str"]).zfill(10)}
        for v in r.json().values()
    ])

@st.cache_data(ttl=3600)
def get_submissions(cik):
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=SEC_HEADERS, timeout=30)
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
    url = (f"https://www.sec.gov/Archives/edgar/data/"
           f"{int(cik)}/{accession.replace('-', '')}/{document}")
    r = requests.get(url, headers=SEC_HEADERS, timeout=40)
    r.raise_for_status()
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
    return text, url

# ============================================================
# XBRL HELPERS
# ============================================================

TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet"
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "receivables": [
        "AccountsReceivableNetCurrent", "AccountsReceivableNet",
        "AccountsNotesAndLoansReceivableNetCurrent"
    ],
    "inventory": [
        "InventoryNet",
        "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings"
    ],
    "debt": [
        "LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent"
    ],
}

def extract_annual(facts, metric):
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in TAGS[metric]:
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
                "val": value
            })

        if rows:
            return (pd.DataFrame(rows)
                    .sort_values(["fy", "end"])
                    .drop_duplicates("fy", keep="last"))
    return pd.DataFrame(columns=["fy", "end", "val"])

def value_for_fy(facts, metric, fy):
    df = extract_annual(facts, metric)
    if df.empty:
        return None
    x = df[df["fy"] == fy]
    return None if x.empty else float(x.iloc[-1]["val"])

def pct_growth(old, new):
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old) * 100

# ============================================================
# TRANSPARENT SCORING
# ============================================================

def score_year(facts, fy):
    metrics = {}
    points = 0
    flags = []

    rev0 = value_for_fy(facts, "revenue", fy - 1)
    rev1 = value_for_fy(facts, "revenue", fy)
    rec0 = value_for_fy(facts, "receivables", fy - 1)
    rec1 = value_for_fy(facts, "receivables", fy)
    ni1 = value_for_fy(facts, "net_income", fy)
    ocf1 = value_for_fy(facts, "ocf", fy)
    debt0 = value_for_fy(facts, "debt", fy - 1)
    debt1 = value_for_fy(facts, "debt", fy)
    inv0 = value_for_fy(facts, "inventory", fy - 1)
    inv1 = value_for_fy(facts, "inventory", fy)

    if rev0 is not None and rev1 is not None:
        metrics["Revenue growth"] = pct_growth(rev0, rev1)

    if rec0 is not None and rec1 is not None:
        metrics["Receivables growth"] = pct_growth(rec0, rec1)

    if metrics.get("Revenue growth") is not None and metrics.get("Receivables growth") is not None:
        gap = metrics["Receivables growth"] - metrics["Revenue growth"]
        if gap >= 15:
            points += 12
            flags.append(("Working capital",
                          "Receivables materially outpaced revenue",
                          12, f"Gap: {gap:.1f} percentage points"))

    if ni1 is not None and ocf1 is not None and ni1 > 0 and ocf1 < ni1 * 0.60:
        ratio = ocf1 / ni1 * 100
        points += 12
        flags.append(("Cash flow",
                      "Operating cash flow materially trailed net income",
                      12, f"OCF / net income: {ratio:.0f}%"))

    if debt0 is not None and debt1 is not None:
        metrics["Debt growth"] = pct_growth(debt0, debt1)
        if metrics["Debt growth"] is not None and metrics["Debt growth"] >= 20:
            points += 8
            flags.append(("Liquidity", "Debt increased materially", 8,
                          f"Debt growth: {metrics['Debt growth']:.1f}%"))

    if inv0 is not None and inv1 is not None:
        metrics["Inventory growth"] = pct_growth(inv0, inv1)
        if metrics["Inventory growth"] is not None and metrics["Inventory growth"] >= 25:
            points += 6
            flags.append(("Business quality", "Inventory increased materially", 6,
                          f"Inventory growth: {metrics['Inventory growth']:.1f}%"))

    score = max(0, 100 - min(60, points))
    return score, flags, metrics

# ============================================================
# FILING SIGNALS
# ============================================================

RULES = [
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
]

def filing_signals(text):
    t = text.lower()
    hits = []
    evidence = []
    for category, label, terms, severity in RULES:
        term = next((term for term in terms if term in t), None)
        if term:
            hits.append({
                "Category": category, "Signal": label,
                "Severity": severity, "Trigger": term
            })
            i = t.find(term)
            evidence.append((label, text[max(0, i-260):i+520]))
    return hits, evidence[:8]

# ============================================================
# HISTORICAL BACKTEST
# ============================================================

SAMPLE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","MA",
    "WMT","COST","KO","NFLX","DIS","ORCL","PEP","ADBE","CRM","INTC"
]

def annual_10ks(subs):
    recent = pd.DataFrame(subs["filings"]["recent"])
    if recent.empty:
        return pd.DataFrame()
    return recent[recent["form"] == "10-K"].sort_values("filingDate")

def historical_observation(ticker, row, facts, cik):
    fy = int(row.get("reportDate", "")[:4]) if row.get("reportDate") else None
    if not fy:
        return None

    score, flags, metrics = score_year(facts, fy)
    next_rev = value_for_fy(facts, "revenue", fy + 1)
    current_rev = value_for_fy(facts, "revenue", fy)

    forward_revenue_growth = pct_growth(current_rev, next_rev)

    # The filing-language signals are not included in the score.
    # They are retained as contextual qualitative evidence.
    return {
        "Ticker": ticker,
        "Fiscal year scored": fy,
        "Filing date": row.get("filingDate", ""),
        "Score": score,
        "Quantitative flags": len(flags),
        "Forward revenue growth": forward_revenue_growth,
        "Revenue growth at score date": metrics.get("Revenue growth"),
    }

def run_backtest(tickers_df, years_per_company=5):
    rows = []
    progress = st.progress(0, text="Building historical observations...")

    for i, ticker in enumerate(SAMPLE):
        m = tickers_df[tickers_df["ticker"] == ticker]

        if m.empty:
            continue

        r = m.iloc[0]

        try:
            subs = get_submissions(r["cik"])
            ks = annual_10ks(subs)

            if ks.empty:
                continue

            facts = get_companyfacts(r["cik"])

            # Find fiscal years for which the following year's
            # revenue is available.
            candidates = []

            for _, row in ks.iterrows():
                if not row.get("reportDate"):
                    continue

                try:
                    fy = int(row["reportDate"][:4])
                except Exception:
                    continue

                if value_for_fy(facts, "revenue", fy + 1) is not None:
                    candidates.append(row)

            # Keep the most recent N eligible fiscal years.
            candidates = candidates[-years_per_company:]

            for row in candidates:
                obs = historical_observation(
                    ticker, row, facts, r["cik"]
                )

                if obs:
                    rows.append(obs)

        except Exception:
            pass

        progress.progress(
            (i + 1) / len(SAMPLE),
            text=f"Processing {ticker} ({i+1}/{len(SAMPLE)})..."
        )

        time.sleep(0.1)

    progress.empty()
    return pd.DataFrame(rows)
# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="SEC Filing Red-Flag Scanner",
                   page_icon="🔎", layout="wide")

st.title("🔎 SEC Filing Red-Flag Scanner")
st.caption("SEC EDGAR → XBRL financial data → transparent risk framework → historical test")

tab1, tab2, tab3 = st.tabs([
    "🔎 Company Scanner", "📊 Cross-Company Research", "🧪 Historical Backtest"
])

# ---------------- Company Scanner ----------------
with tab1:
    try:
        tickers = get_tickers()
    except Exception as e:
        st.error(f"SEC company list unavailable: {e}")
        st.stop()

    q = st.text_input("Search company", placeholder="Apple, NVIDIA, Tesla...")
    matches = tickers
    if q:
        matches = tickers[
            tickers["ticker"].str.contains(q.upper(), na=False) |
            tickers["name"].str.contains(q, case=False, na=False)
        ]

    matches = matches.head(25)
    options = {
        f"{r.ticker} — {r.name}": (r.ticker, r.cik, r.name)
        for _, r in matches.iterrows()
    }

    choice = st.selectbox("Company",
                          list(options) if options else ["No matches"])

    if st.button("Run red-flag scan", type="primary"):
        ticker, cik, company = options[choice]
        try:
            with st.spinner("Reading SEC filing and XBRL data..."):
                subs = get_submissions(cik)
                ks = annual_10ks(subs)
                if ks.empty:
                    st.error("No recent 10-K found.")
                    st.stop()

                latest = ks.iloc[-1]
                facts = get_companyfacts(cik)
                fy = int(latest["reportDate"][:4])

                score, qflags, metrics = score_year(facts, fy)
                text, url = get_filing(cik, latest["accessionNumber"],
                                        latest["primaryDocument"])
                review, evidence = filing_signals(text)

                level = ("Low" if score >= 85 else
                         "Moderate" if score >= 70 else
                         "Elevated" if score >= 50 else "High")

                st.session_state["result"] = {
                    "company": company, "ticker": ticker, "score": score,
                    "level": level, "qflags": qflags, "review": review,
                    "evidence": evidence, "metrics": metrics,
                    "filing_date": latest["filingDate"], "url": url
                }
        except Exception as e:
            st.error(f"SEC analysis failed: {e}")

    if "result" in st.session_state:
        x = st.session_state["result"]

        a,b,c = st.columns(3)
        a.metric("Financial profile score", f"{x['score']}/100")
        b.metric("Risk level", x["level"])
        c.metric("Quantitative flags", len(x["qflags"]))

        st.caption(
            "Higher score = stronger profile. Filing-language matches are review "
            "signals and are intentionally excluded from the quantitative score."
        )

        st.subheader("Quantitative risk signals")
        if x["qflags"]:
            st.dataframe(pd.DataFrame([
                {"Category": a, "Signal": b, "Points": c, "Evidence": d}
                for a,b,c,d in x["qflags"]
            ]), use_container_width=True, hide_index=True)
        else:
            st.success("No quantitative thresholds were triggered.")

        st.subheader("Filing-language review signals")
        if x["review"]:
            st.dataframe(pd.DataFrame(x["review"]),
                         use_container_width=True, hide_index=True)
        else:
            st.success("No predefined review signals detected.")

        st.subheader("Evidence from the filing")
        for i,(label,snip) in enumerate(x["evidence"],1):
            with st.expander(f"{i}. {label}"):
                st.write(snip)

        st.subheader("Analyst follow-ups")
        for item in [
            "Reconcile reported earnings with operating/free cash flow.",
            "Investigate working-capital movements relative to revenue.",
            "Inspect debt maturities, covenants and refinancing needs.",
            "Read the relevant footnotes and risk factors in context.",
        ]:
            st.write("• " + item)

        st.markdown(f"[Open SEC filing]({x['url']})")

# ---------------- Cross-company ----------------
with tab2:
    st.subheader("📊 Cross-Company Research Study")
    st.write(
        "**Research question:** Can a transparent, automated financial-risk "
        "framework identify differences in financial profiles across major public companies?"
    )
    st.info(
        "The current 20-company sample is descriptive. The score is not yet "
        "calibrated as a probability of distress."
    )

    if st.button("Run 20-company study", type="primary"):
        try:
            study = []
            tickers = get_tickers()
            progress = st.progress(0, text="Analyzing companies...")
            for i,ticker in enumerate(SAMPLE):
                m=tickers[tickers["ticker"]==ticker]
                if m.empty:
                    continue
                r=m.iloc[0]
                try:
                    facts=get_companyfacts(r["cik"])
                    subs=get_submissions(r["cik"])
                    ks=annual_10ks(subs)
                    if ks.empty:
                        continue
                    fy=int(ks.iloc[-1]["reportDate"][:4])
                    score,flags,_=score_year(facts,fy)
                    review,_=filing_signals(
                        get_filing(r["cik"],ks.iloc[-1]["accessionNumber"],
                                   ks.iloc[-1]["primaryDocument"])[0]
                    )
                    study.append({
                        "Ticker":ticker,"Company":r["name"],
                        "Risk score":score,
                        "Risk level":("Low" if score>=85 else "Moderate" if score>=70
                                      else "Elevated" if score>=50 else "High"),
                        "Quantitative flags":len(flags),
                        "Filing review signals":len(review)
                    })
                except Exception:
                    pass
                progress.progress((i+1)/len(SAMPLE))
            progress.empty()
            st.session_state["study"]=pd.DataFrame(study)
        except Exception as e:
            st.error(f"Study failed: {e}")

    if "study" in st.session_state and not st.session_state["study"].empty:
        d=st.session_state["study"].sort_values("Risk score")
        a,b,c=st.columns(3)
        a.metric("Companies analyzed",len(d))
        b.metric("Average score",f"{d['Risk score'].mean():.1f}/100")
        c.metric("Scores below 85",(d["Risk score"]<85).sum())

        st.dataframe(d,use_container_width=True,hide_index=True)
        st.subheader("Risk score distribution")
        st.bar_chart(d.set_index("Ticker")[["Risk score"]])

        st.caption(
            "Important: this is a descriptive cross-section. Because most companies "
            "score highly, the next research step is historical calibration."
        )

# ---------------- Historical Backtest ----------------
with tab3:
    st.subheader("🧪 Historical Backtest")
    st.write(
        """
        This test asks a stronger question:

        **When the framework is applied to an earlier fiscal year, does the resulting
        score relate to the company's subsequent revenue growth?**

        The framework is scored using information available for the selected fiscal
        year. The outcome is the following year's reported revenue growth.
        """
    )

    st.warning(
        "A 20-company backtest is exploratory, not statistically conclusive. "
        "Its purpose is to test the framework and generate a falsifiable research hypothesis."
    )

    if st.button("Run historical backtest", type="primary"):
        try:
            tickers=get_tickers()
            bt=run_backtest(tickers)
            st.session_state["backtest"]=bt
        except Exception as e:
            st.error(f"Backtest failed: {e}")

    if "backtest" in st.session_state and not st.session_state["backtest"].empty:
        bt=st.session_state["backtest"].dropna(
            subset=["Score","Forward revenue growth"]
        ).copy()

        st.metric("Historical observations",len(bt))
        st.dataframe(bt,use_container_width=True,hide_index=True)
        if len(bt)>=3:
            corr=bt["Score"].corr(bt["Forward revenue growth"])
            st.metric(
                "Score vs. next-year revenue-growth correlation",
                "Not available" if pd.isna(corr) else f"{corr:.2f}"
            )

            st.subheader("Framework score vs. subsequent revenue growth")

            st.scatter_chart(
                bt,
                x="Score",
                y="Forward revenue growth",
                x_label="Historical framework score",
                y_label="Following-year revenue growth (%)"
            )

            if not pd.isna(corr):
                direction = "positive" if corr > 0 else "negative"
                st.write(
                    f"In this exploratory sample, the correlation is **{corr:.2f}** "
                    f"({direction}). This is an association, not evidence of causation "
                    f"or investment performance."
                )
 

        st.download_button(
            "Download historical backtest (CSV)",
            bt.to_csv(index=False),
            "sec_red_flag_historical_backtest.csv",
            "text/csv"
        )

    st.markdown("---")
    st.subheader("Methodology")
    st.markdown(
        """
**Quantitative scoring**

- Receivables growth ≥ 15 percentage points above revenue growth: 12 points
- Operating cash flow < 60% of net income: 12 points
- Debt growth ≥ 20%: 8 points
- Inventory growth ≥ 25%: 6 points

The score starts at 100 and subtracts triggered points. Filing-language signals are
kept separate because keyword presence alone does not establish financial distress.

**Historical test**

For each company, an earlier 10-K fiscal year is selected when the following year's
revenue is available. The framework score is calculated for the earlier year and
compared with subsequent reported revenue growth. This creates a simple out-of-sample
directional test rather than assuming the framework works.
"""
    )
