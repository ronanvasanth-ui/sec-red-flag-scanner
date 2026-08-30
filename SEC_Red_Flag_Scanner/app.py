import re
import time
import pandas as pd
import requests
import streamlit as st

SEC_HEADERS = {
    "User-Agent": "SEC Filing Red-Flag Scanner/1.0 academic-research-demo contact@example.com"
}

# ============================================================
# SEC DATA
# ============================================================

@st.cache_data(ttl=3600)
def get_tickers():
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=20
    )
    r.raise_for_status()
    return pd.DataFrame([
        {
            "ticker": v["ticker"],
            "name": v["title"],
            "cik": str(v["cik_str"]).zfill(10)
        }
        for v in r.json().values()
    ])


@st.cache_data(ttl=3600)
def get_submissions(cik):
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=30
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def get_companyfacts(cik):
    r = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=30
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


# ============================================================
# XBRL HELPERS
# ============================================================

TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet"
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss"
    ],
    "ocf": [
        "NetCashProvidedByUsedInOperatingActivities"
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
        "AccountsNotesAndLoansReceivableNetCurrent"
    ],
    "inventory": [
        "InventoryNet",
        "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings"
    ],
    "debt": [
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent"
    ],
}


def extract_annual(facts, metric):
    """Extract one defensible annual observation per fiscal year.

    Duration-based metrics (revenue, net income and OCF) are restricted to
    roughly annual reporting periods. Instant/balance-sheet metrics are not.
    This avoids accidentally selecting a short-period XBRL fact that carries
    the same fiscal-year label.
    """
    usgaap = facts.get("facts", {}).get("us-gaap", {})

    duration_metrics = {"revenue", "net_income", "ocf"}

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
                fy = int(x["fy"])
            except Exception:
                continue

            start = x.get("start")
            end = x.get("end")

            # For duration metrics, require a roughly full fiscal-year period.
            if metric in duration_metrics:
                if not start or not end:
                    continue

                try:
                    start_dt = pd.Timestamp(start)
                    end_dt = pd.Timestamp(end)
                    days = (end_dt - start_dt).days
                except Exception:
                    continue

                if not 300 <= days <= 380:
                    continue

            rows.append({
                "fy": fy,
                "start": start,
                "end": end,
                "val": value
            })

        if rows:
            df = pd.DataFrame(rows)

            # Prefer the latest reporting end date for a fiscal year.
            # This makes the selection deterministic when SEC facts contain
            # multiple 10-K observations carrying the same FY.
            return (
                df.sort_values(["fy", "end"])
                .drop_duplicates("fy", keep="last")
                .reset_index(drop=True)
            )

    return pd.DataFrame(columns=["fy", "start", "end", "val"])


def value_for_fy(facts, metric, fy):
    df = extract_annual(facts, metric)

    if df.empty:
        return None

    x = df[df["fy"] == fy]

    if x.empty:
        return None

    return float(x.iloc[-1]["val"])


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

    if (
        metrics.get("Revenue growth") is not None
        and metrics.get("Receivables growth") is not None
    ):
        gap = (
            metrics["Receivables growth"]
            - metrics["Revenue growth"]
        )

        if gap >= 15:
            points += 12
            flags.append((
                "Working capital",
                "Receivables materially outpaced revenue",
                12,
                f"Gap: {gap:.1f} percentage points"
            ))

    if (
        ni1 is not None
        and ocf1 is not None
        and ni1 > 0
        and ocf1 < ni1 * 0.60
    ):
        ratio = ocf1 / ni1 * 100
        points += 12

        flags.append((
            "Cash flow",
            "Operating cash flow materially trailed net income",
            12,
            f"OCF / net income: {ratio:.0f}%"
        ))

    if debt0 is not None and debt1 is not None:
        metrics["Debt growth"] = pct_growth(debt0, debt1)

        if (
            metrics["Debt growth"] is not None
            and metrics["Debt growth"] >= 20
        ):
            points += 8

            flags.append((
                "Liquidity",
                "Debt increased materially",
                8,
                f"Debt growth: {metrics['Debt growth']:.1f}%"
            ))

    if inv0 is not None and inv1 is not None:
        metrics["Inventory growth"] = pct_growth(inv0, inv1)

        if (
            metrics["Inventory growth"] is not None
            and metrics["Inventory growth"] >= 25
        ):
            points += 6

            flags.append((
                "Business quality",
                "Inventory increased materially",
                6,
                f"Inventory growth: {metrics['Inventory growth']:.1f}%"
            ))

    score = max(0, 100 - min(60, points))

    return score, flags, metrics


def risk_level(score):
    if score >= 85:
        return "Low"
    if score >= 70:
        return "Moderate"
    if score >= 50:
        return "Elevated"
    return "High"


# ============================================================
# FILING SIGNALS
# ============================================================

RULES = [
    (
        "Accounting / controls",
        "Material weakness language",
        ["material weakness"],
        3
    ),
    (
        "Accounting / controls",
        "Restatement language",
        ["restatement", "restated financial statements"],
        3
    ),
    (
        "Liquidity",
        "Going-concern language",
        ["going concern", "substantial doubt about"],
        4
    ),
    (
        "Liquidity",
        "Debt covenant language",
        ["debt covenant", "covenant violation"],
        2
    ),
    (
        "Governance",
        "Related-party language",
        ["related party", "related-party"],
        1
    ),
    (
        "Business quality",
        "Customer concentration language",
        ["customer concentration", "concentration of customers"],
        1
    ),
    (
        "Business quality",
        "Impairment language",
        ["impairment charge"],
        1
    ),
    (
        "Business quality",
        "Restructuring language",
        ["restructuring charge"],
        1
    ),
]


def filing_signals(text):
    t = text.lower()
    hits = []
    evidence = []

    for category, label, terms, severity in RULES:
        term = next((term for term in terms if term in t), None)

        if term:
            hits.append({
                "Category": category,
                "Signal": label,
                "Severity": severity,
                "Trigger": term
            })

            i = t.find(term)
            evidence.append((
                label,
                text[max(0, i - 260):i + 520]
            ))

    return hits, evidence[:8]


# ============================================================
# RESEARCH / DISPLAY HELPERS
# ============================================================

def correlation_label(value):
    if pd.isna(value):
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


def score_range_summary(bt):
    frame = bt.copy()
    frame["Score range"] = pd.cut(
        frame["Score"],
        bins=[-1, 69, 84, 100],
        labels=[
            "High risk (<70)",
            "Moderate (70–84)",
            "Low risk (85–100)"
        ]
    )
    summary = (
        frame.groupby("Score range", observed=False)["Forward revenue growth"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary.columns = [
        "Score range",
        "Observations",
        "Mean forward growth (%)",
        "Median forward growth (%)"
    ]
    return summary


def research_note(title, body, kind="info"):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(body)


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "MA",
    "WMT", "COST", "KO", "NFLX", "DIS",
    "ORCL", "PEP", "ADBE", "CRM", "INTC"
]


def annual_10ks(subs):
    recent = pd.DataFrame(subs["filings"]["recent"])

    if recent.empty:
        return pd.DataFrame()

    return recent[
        recent["form"] == "10-K"
    ].sort_values("filingDate")


def historical_observation(ticker, row, facts, cik):
    fy = (
        int(row.get("reportDate", "")[:4])
        if row.get("reportDate")
        else None
    )

    if not fy:
        return None

    score, flags, metrics = score_year(facts, fy)

    next_rev = value_for_fy(facts, "revenue", fy + 1)
    current_rev = value_for_fy(facts, "revenue", fy)

    forward_revenue_growth = pct_growth(
        current_rev,
        next_rev
    )

    return {
        "Ticker": ticker,
        "Fiscal year scored": fy,
        "Filing date": row.get("filingDate", ""),
        "Score": score,
        "Risk level": risk_level(score),
        "Quantitative flags": len(flags),
        "Forward revenue growth": forward_revenue_growth,
        "Revenue growth at score date": metrics.get(
            "Revenue growth"
        ),
    }


def run_backtest(tickers_df, years_per_company=5):
    rows = []

    progress = st.progress(
        0,
        text="Building historical observations..."
    )

    for i, ticker in enumerate(SAMPLE):
        m = tickers_df[
            tickers_df["ticker"] == ticker
        ]

        if m.empty:
            continue

        r = m.iloc[0]

        try:
            subs = get_submissions(r["cik"])
            ks = annual_10ks(subs)

            if ks.empty:
                continue

            facts = get_companyfacts(r["cik"])

            candidates = []

            for _, row in ks.iterrows():
                if not row.get("reportDate"):
                    continue

                try:
                    fy = int(row["reportDate"][:4])
                except Exception:
                    continue

                if value_for_fy(
                    facts,
                    "revenue",
                    fy + 1
                ) is not None:
                    candidates.append(row)

            candidates = candidates[-years_per_company:]

            for row in candidates:
                obs = historical_observation(
                    ticker,
                    row,
                    facts,
                    r["cik"]
                )

                if obs:
                    rows.append(obs)

        except Exception:
            pass

        progress.progress(
            (i + 1) / len(SAMPLE),
            text=f"Processing {ticker} ({i + 1}/{len(SAMPLE)})..."
        )

        time.sleep(0.1)

    progress.empty()

    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="SEC Filing Red-Flag Scanner",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 SEC Filing Red-Flag Scanner")
st.caption(
    "SEC EDGAR → XBRL financial data → transparent financial-profile framework → historical test"
)
st.info(
    "Research design: quantitative filing data are scored separately from qualitative filing-language review signals. "
    "The historical test is exploratory and is not investment advice."
)

tab1, tab2, tab3 = st.tabs([
    "🔎 Company Scanner",
    "📊 Cross-Company Research",
    "🧪 Historical Backtest"
])


# ---------------- Company Scanner ----------------

with tab1:
    try:
        tickers = get_tickers()
    except Exception as e:
        st.error(f"SEC company list unavailable: {e}")
        st.stop()

    q = st.text_input(
        "Search company",
        placeholder="Apple, NVIDIA, Tesla..."
    )

    matches = tickers

    if q:
        matches = tickers[
            tickers["ticker"].str.contains(
                q.upper(),
                na=False
            )
            |
            tickers["name"].str.contains(
                q,
                case=False,
                na=False
            )
        ]

    matches = matches.head(25)

    options = {
        f"{r.ticker} — {r.name}": (
            r.ticker,
            r.cik,
            r.name
        )
        for _, r in matches.iterrows()
    }

    choice = st.selectbox(
        "Company",
        list(options) if options else ["No matches"]
    )

    if st.button(
        "Run red-flag scan",
        type="primary"
    ):
        ticker, cik, company = options[choice]

        try:
            with st.spinner(
                "Reading SEC filing and XBRL data..."
            ):
                subs = get_submissions(cik)
                ks = annual_10ks(subs)

                if ks.empty:
                    st.error("No recent 10-K found.")
                    st.stop()

                latest = ks.iloc[-1]
                facts = get_companyfacts(cik)
                fy = int(latest["reportDate"][:4])

                score, qflags, metrics = score_year(
                    facts,
                    fy
                )

                text, url = get_filing(
                    cik,
                    latest["accessionNumber"],
                    latest["primaryDocument"]
                )

                review, evidence = filing_signals(text)

                st.session_state["result"] = {
                    "company": company,
                    "ticker": ticker,
                    "score": score,
                    "level": risk_level(score),
                    "qflags": qflags,
                    "review": review,
                    "evidence": evidence,
                    "metrics": metrics,
                    "filing_date": latest["filingDate"],
                    "url": url
                }

        except Exception as e:
            st.error(f"SEC analysis failed: {e}")

    if "result" in st.session_state:
        x = st.session_state["result"]

        a, b, c = st.columns(3)

        a.metric(
            "Financial profile score",
            f"{x['score']}/100"
        )

        b.metric(
            "Risk level",
            x["level"]
        )

        c.metric(
            "Quantitative flags",
            len(x["qflags"])
        )

        st.caption(f"Latest 10-K filing: {x['filing_date']}")

        st.caption(
            "Higher score = stronger financial profile. Filing-language matches are "
            "review signals and are intentionally excluded from the quantitative score."
        )

        with st.container(border=True):
            st.markdown("**How to read this score**")
            st.write(
                "The score starts at 100 and subtracts points only when predefined "
                "quantitative thresholds are triggered. A high score means fewer "
                "detected quantitative warning conditions; it does not mean the "
                "company is free of risk."
            )

        st.subheader("Quantitative risk signals")

        if x["qflags"]:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Category": a,
                        "Signal": b,
                        "Points": c,
                        "Evidence": d
                    }
                    for a, b, c, d in x["qflags"]
                ]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(
                "No quantitative thresholds were triggered."
            )

        st.subheader("Filing-language review signals")

        if x["review"]:
            st.dataframe(
                pd.DataFrame(x["review"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(
                "No predefined review signals detected."
            )

        st.subheader("Evidence from the filing")

        for i, (label, snip) in enumerate(
            x["evidence"],
            1
        ):
            with st.expander(f"{i}. {label}"):
                st.write(snip)

        st.subheader("Analyst follow-ups")

        for item in [
            "Reconcile reported earnings with operating/free cash flow.",
            "Investigate working-capital movements relative to revenue.",
            "Inspect debt maturities, covenants and refinancing needs.",
            "Read the relevant footnotes and risk factors in context."
        ]:
            st.write("• " + item)

        st.markdown(
            f"[Open SEC filing]({x['url']})"
        )


# ---------------- Cross-company ----------------

with tab2:
    st.subheader("📊 Cross-Company Research Study")

    st.write(
        "**Research question:** Can a transparent, automated financial-risk "
        "framework identify meaningful differences in financial profiles "
        "across major public companies?"
    )

    st.info(
        "The sample is intentionally fixed at 20 large public companies so the methodology is "
        "reproducible rather than cherry-picked. The cross-sectional result is descriptive: "
        "it is not a representative sample of all public companies."
    )

    if st.button(
        "Run 20-company study",
        type="primary"
    ):
        try:
            study = []
            tickers = get_tickers()

            progress = st.progress(
                0,
                text="Analyzing companies..."
            )

            for i, ticker in enumerate(SAMPLE):
                m = tickers[
                    tickers["ticker"] == ticker
                ]

                if m.empty:
                    continue

                r = m.iloc[0]

                try:
                    facts = get_companyfacts(r["cik"])
                    subs = get_submissions(r["cik"])
                    ks = annual_10ks(subs)

                    if ks.empty:
                        continue

                    latest = ks.iloc[-1]
                    fy = int(latest["reportDate"][:4])

                    score, flags, _ = score_year(
                        facts,
                        fy
                    )

                    filing_text, _ = get_filing(
                        r["cik"],
                        latest["accessionNumber"],
                        latest["primaryDocument"]
                    )

                    review, _ = filing_signals(
                        filing_text
                    )

                    study.append({
                        "Ticker": ticker,
                        "Company": r["name"],
                        "Risk score": score,
                        "Risk level": risk_level(score),
                        "Quantitative flags": len(flags),
                        "Filing review signals": len(review)
                    })

                except Exception:
                    pass

                progress.progress(
                    (i + 1) / len(SAMPLE),
                    text=f"Processing {ticker} ({i + 1}/{len(SAMPLE)})..."
                )

            progress.empty()

            st.session_state["study"] = pd.DataFrame(
                study
            )

        except Exception as e:
            st.error(f"Study failed: {e}")

    if (
        "study" in st.session_state
        and not st.session_state["study"].empty
    ):
        d = st.session_state["study"].sort_values(
            ["Risk score", "Ticker"],
            ascending=[True, True]
        ).reset_index(drop=True)

        a, b, c = st.columns(3)

        a.metric(
            "Companies analyzed",
            len(d)
        )

        b.metric(
            "Average score",
            f"{d['Risk score'].mean():.1f}/100"
        )

        c.metric(
            "Elevated/high-risk profiles",
            int((d["Risk score"] < 70).sum())
        )

        st.dataframe(
            d,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Risk score distribution")

        st.bar_chart(
            d.set_index("Ticker")[["Risk score"]]
        )

        st.download_button(
            "Download research dataset (CSV)",
            d.to_csv(index=False),
            "sec_red_flag_cross_company_research.csv",
            "text/csv"
        )

        low_variation = d["Risk score"].nunique() <= 4
        if low_variation or (d["Risk score"] >= 85).mean() >= 0.75:
            st.warning(
                "Interpretation: most companies cluster in the high-score range. "
                "That limited variation makes this cross-sectional comparison a weak "
                "test of discrimination; the historical backtest is the stronger calibration test."
            )
        else:
            st.info(
                "Interpretation: the sample contains more score variation, but this remains "
                "descriptive cross-sectional research rather than predictive evidence."
            )

        research_note(
            "Research takeaway",
            "Use this table to compare detected warning conditions across the fixed sample, not to rank companies as investments. "
            "Filing-language signals are contextual and are not added to the quantitative score."
        )


# ---------------- Historical Backtest ----------------

with tab3:
    st.subheader("🧪 Historical Backtest")

    st.write(
        """
        This test asks a stronger question:

        **When the framework is applied to an earlier fiscal year, does the
        resulting score relate to the company's subsequent revenue growth?**

        The framework is scored using information available for the selected
        fiscal year. The outcome is the following year's reported revenue growth.
        """
    )

    st.warning(
        "A 20-company backtest is exploratory, not statistically conclusive. Its purpose is to test the framework "
        "and generate a falsifiable research hypothesis. A correlation here does not establish causation or investment performance."
    )

    if st.button(
        "Run historical backtest",
        type="primary"
    ):
        try:
            tickers = get_tickers()

            bt = run_backtest(
                tickers,
                years_per_company=5
            )

            st.session_state["backtest"] = bt

        except Exception as e:
            st.error(f"Backtest failed: {e}")

    if (
        "backtest" in st.session_state
        and not st.session_state["backtest"].empty
    ):
        bt = st.session_state["backtest"].dropna(
            subset=[
                "Score",
                "Forward revenue growth"
            ]
        ).copy()

        st.metric(
            "Historical observations",
            len(bt)
        )

        st.caption(
            "Annual XBRL duration filters are applied to revenue, net income "
            "and operating cash flow to reduce distorted period selections."
        )

        st.dataframe(
            bt,
            use_container_width=True,
            hide_index=True
        )

        if len(bt) >= 3:
            pearson = bt["Score"].corr(
                bt["Forward revenue growth"],
                method="pearson"
            )

            # Spearman is calculated from ranks so scipy is not required.
            score_rank = bt["Score"].rank(method="average")
            growth_rank = bt["Forward revenue growth"].rank(method="average")
            spearman = score_rank.corr(growth_rank)

            a, b, c = st.columns(3)
            a.metric(
                "Pearson correlation",
                "Not available" if pd.isna(pearson) else f"{pearson:.2f}"
            )
            b.metric(
                "Spearman correlation",
                "Not available" if pd.isna(spearman) else f"{spearman:.2f}"
            )
            c.metric(
                "Unique score values",
                int(bt["Score"].nunique())
            )

            st.subheader("Key finding")
            if pd.isna(pearson):
                st.info(
                    "The sample does not contain enough variation to calculate a meaningful Pearson correlation."
                )
            else:
                label = correlation_label(pearson)
                st.info(
                    f"In this exploratory sample, the Pearson correlation is **{pearson:.2f}**, indicating a "
                    f"**{label}** between the framework score and following-year revenue growth. "
                    "This does not establish causation, predictive power, or investment performance."
                )

            if not pd.isna(spearman):
                st.caption(
                    f"Spearman ρ = {spearman:.2f}. The rank-based result is useful as a robustness check because it is "
                    "less dependent on the exact scale of individual observations."
                )

            st.subheader("Framework score vs. subsequent revenue growth")
            st.scatter_chart(
                bt,
                x="Score",
                y="Forward revenue growth",
                x_label="Historical framework score",
                y_label="Following-year revenue growth (%)"
            )

            summary = score_range_summary(bt)
            st.subheader("Subsequent growth by score range")
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            low_n = int(summary.loc[summary["Score range"] == "Low risk (85–100)", "Observations"].iloc[0]) if not summary.loc[summary["Score range"] == "Low risk (85–100)"].empty else 0
            moderate_n = int(summary.loc[summary["Score range"] == "Moderate (70–84)", "Observations"].iloc[0]) if not summary.loc[summary["Score range"] == "Moderate (70–84)"].empty else 0
            high_n = int(summary.loc[summary["Score range"] == "High risk (<70)", "Observations"].iloc[0]) if not summary.loc[summary["Score range"] == "High risk (<70)"].empty else 0

            if max(low_n, moderate_n, high_n) > 0.75 * len(bt):
                st.warning(
                    f"Sample-balance limitation: {max(low_n, moderate_n, high_n)} of {len(bt)} observations fall in one score range. "
                    "Comparisons between ranges should therefore be treated cautiously."
                )

            st.caption(
                "Score-range results are descriptive. The backtest tests association with subsequent reported revenue growth, "
                "not stock returns, valuation, fraud, or causation."
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
        "**Core design:** 100-point quantitative financial-profile score + separate qualitative filing review. "
        "The quantitative score uses four predefined thresholds; qualitative keywords are shown as review signals only."
    )

    st.markdown(
        """
**Quantitative scoring**

- Receivables growth ≥ 15 percentage points above revenue growth: 12 points
- Operating cash flow < 60% of net income: 12 points
- Debt growth ≥ 20%: 8 points
- Inventory growth ≥ 25%: 6 points

The score starts at 100 and subtracts triggered points. Filing-language
signals are kept separate because keyword presence alone does not establish
financial distress.

**Historical test**

For each company, an earlier 10-K fiscal year is selected when the following
year's revenue is available. The framework score is calculated for the earlier
year and compared with subsequent reported revenue growth. This creates a
simple out-of-sample directional test rather than assuming the framework works.

**Correlation measures**

Pearson correlation measures linear association between score and subsequent
growth. Spearman correlation measures rank-based association and is less
sensitive to the exact scale of individual observations.

The backtest is exploratory and is not evidence of causation, future stock
returns, or investment performance.
"""
    )
