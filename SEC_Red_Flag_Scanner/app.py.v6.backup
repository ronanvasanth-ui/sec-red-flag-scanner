
import re
import time
import pandas as pd
import requests
import streamlit as st

# ============================================================
# SEC Filing Red-Flag Scanner — clean V11
# ============================================================

SEC_HEADERS = {
    "User-Agent": "SEC Filing Red-Flag Scanner research demo contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

ANNUAL_FORMS = {"10-K", "20-F", "40-F"}

SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "MA",
    "WMT", "COST", "KO", "NFLX", "DIS",
    "ORCL", "PEP", "ADBE", "CRM", "INTC",
]

# ---------------- SEC data ----------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_tickers():
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return pd.DataFrame(
        [
            {
                "ticker": str(v["ticker"]).upper(),
                "name": v["title"],
                "cik": str(v["cik_str"]).zfill(10),
            }
            for v in r.json().values()
        ]
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_submissions(cik):
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_companyfacts(cik):
    r = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def get_filing(cik, accession, document):
    accession_clean = str(accession).replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_clean}/{document}"
    )
    r = requests.get(url, headers=SEC_HEADERS, timeout=45)
    r.raise_for_status()
    text = r.text
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text, url


# ---------------- Filing helpers ----------------

def annual_reports(subs):
    recent = pd.DataFrame(subs.get("filings", {}).get("recent", {}))
    if recent.empty or "form" not in recent.columns:
        return pd.DataFrame()

    out = recent[recent["form"].isin(ANNUAL_FORMS)].copy()
    if out.empty:
        return out

    if "filingDate" in out.columns:
        out = out.sort_values("filingDate").reset_index(drop=True)
    return out


def latest_annual(subs):
    reports = annual_reports(subs)
    if reports.empty:
        return None
    return reports.iloc[-1]


def filing_form_label(form):
    return {
        "10-K": "10-K",
        "20-F": "20-F",
        "40-F": "40-F",
    }.get(form, str(form))


# ---------------- XBRL ----------------

TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "ProfitLossAttributableToOwnersOfParent",
    ],
    "ocf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromUsedInOperatingActivities",
        "NetCashFlowsFromUsedInOperatingActivities",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
        "AccountsNotesAndLoansReceivableNetCurrent",
        "TradeAndOtherCurrentReceivables",
        "TradeAndOtherReceivables",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryNetOfAllowancesCustomerAdvancesAndProgressBillings",
        "Inventories",
    ],
    "debt": [
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "BorrowingsCurrent",
        "BorrowingsNoncurrent",
        "CurrentBorrowings",
        "NoncurrentBorrowings",
    ],
}

DURATION_METRICS = {"revenue", "net_income", "ocf"}


def extract_annual(facts, metric):
    namespaces = facts.get("facts", {})
    candidates = []

    namespace_order = [
        ("us-gaap", TAGS[metric]),
        ("ifrs-full", TAGS[metric]),
    ]

    for namespace, tag_list in namespace_order:
        concept_store = namespaces.get(namespace, {})

        for tag in tag_list:
            concept = concept_store.get(tag)
            if not concept:
                continue

            units = concept.get("units", {})
            unit = "USD" if "USD" in units else next(iter(units), None)
            if not unit:
                continue

            rows = []
            for x in units.get(unit, []):
                if x.get("form") not in ANNUAL_FORMS or "fy" not in x:
                    continue

                try:
                    value = float(x["val"])
                    fy = int(x["fy"])
                except (TypeError, ValueError):
                    continue

                start = x.get("start")
                end = x.get("end")

                if metric in DURATION_METRICS:
                    if not start or not end:
                        continue
                    try:
                        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
                    except Exception:
                        continue
                    if not 300 <= days <= 380:
                        continue

                rows.append(
                    {
                        "fy": fy,
                        "start": start,
                        "end": end,
                        "val": value,
                        "namespace": namespace,
                        "tag": tag,
                    }
                )

            candidates.extend(rows)

    if not candidates:
        return pd.DataFrame(
            columns=["fy", "start", "end", "val", "namespace", "tag"]
        )

    df = pd.DataFrame(candidates)
    df["gaap_priority"] = (df["namespace"] == "us-gaap").astype(int)

    return (
        df.sort_values(["fy", "end", "gaap_priority"], na_position="first")
        .drop_duplicates("fy", keep="last")
        .reset_index(drop=True)
    )


def value_for_fy(facts, metric, fy):
    df = extract_annual(facts, metric)
    if df.empty:
        return None

    row = df[df["fy"] == fy]
    if row.empty:
        return None

    return float(row.iloc[-1]["val"])


def pct_growth(old, new):
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old) * 100.0


# ---------------- Quantitative scoring ----------------

def score_year(facts, fy):
    metrics = {}
    points = 0
    flags = []

    rev0 = value_for_fy(facts, "revenue", fy - 1)
    rev1 = value_for_fy(facts, "revenue", fy)

    rec0 = value_for_fy(facts, "receivables", fy - 1)
    rec1 = value_for_fy(facts, "receivables", fy)

    ni0 = value_for_fy(facts, "net_income", fy - 1)
    ni1 = value_for_fy(facts, "net_income", fy)
    ocf1 = value_for_fy(facts, "ocf", fy)

    debt0 = value_for_fy(facts, "debt", fy - 1)
    debt1 = value_for_fy(facts, "debt", fy)

    inv0 = value_for_fy(facts, "inventory", fy - 1)
    inv1 = value_for_fy(facts, "inventory", fy)

    # 1. Revenue decline
    if rev0 is not None and rev1 is not None:
        metrics["Revenue growth"] = pct_growth(rev0, rev1)
        if metrics["Revenue growth"] <= -5:
            points += 6
            flags.append(
                (
                    "Growth",
                    "Revenue declined materially",
                    6,
                    f"Revenue growth: {metrics['Revenue growth']:.1f}%",
                )
            )

    # 2. Receivables growth materially above revenue
    if rec0 is not None and rec1 is not None:
        metrics["Receivables growth"] = pct_growth(rec0, rec1)

    if (
        metrics.get("Revenue growth") is not None
        and metrics.get("Receivables growth") is not None
    ):
        gap = metrics["Receivables growth"] - metrics["Revenue growth"]
        metrics["Receivables / revenue growth gap"] = gap

        if gap >= 10:
            points += 10
            flags.append(
                (
                    "Working capital",
                    "Receivables materially outpaced revenue",
                    10,
                    f"Gap: {gap:.1f} percentage points",
                )
            )

    # 3–4. Cash conversion
    if ni1 is not None and ocf1 is not None:
        if ni1 > 0:
            ratio = ocf1 / ni1 * 100
            metrics["OCF / net income"] = ratio

            if ocf1 < 0:
                points += 12
                flags.append(
                    (
                        "Cash flow",
                        "Operating cash flow was negative despite positive net income",
                        12,
                        f"OCF / net income: {ratio:.0f}%",
                    )
                )
            elif ratio < 80:
                points += 10
                flags.append(
                    (
                        "Cash flow",
                        "Operating cash flow materially trailed net income",
                        10,
                        f"OCF / net income: {ratio:.0f}%",
                    )
                )
        elif ocf1 < 0:
            points += 10
            flags.append(
                (
                    "Cash flow",
                    "Operating cash flow was negative",
                    10,
                    "Negative OCF",
                )
            )

    # 5. Debt growth
    if debt0 is not None and debt1 is not None:
        metrics["Debt growth"] = pct_growth(debt0, debt1)
        if metrics["Debt growth"] >= 10:
            points += 8
            flags.append(
                (
                    "Liquidity",
                    "Debt increased materially",
                    8,
                    f"Debt growth: {metrics['Debt growth']:.1f}%",
                )
            )

    # 6. Inventory growth
    if inv0 is not None and inv1 is not None:
        metrics["Inventory growth"] = pct_growth(inv0, inv1)
        if metrics["Inventory growth"] >= 15:
            points += 6
            flags.append(
                (
                    "Business quality",
                    "Inventory increased materially",
                    6,
                    f"Inventory growth: {metrics['Inventory growth']:.1f}%",
                )
            )

    # 7. Net income decline
    if ni0 is not None and ni1 is not None and ni0 != 0:
        metrics["Net income growth"] = pct_growth(ni0, ni1)
        if metrics["Net income growth"] <= -10:
            points += 6
            flags.append(
                (
                    "Profitability",
                    "Net income declined materially",
                    6,
                    f"Net income growth: {metrics['Net income growth']:.1f}%",
                )
            )

    # Cap deductions so score stays interpretable.
    score = max(0, 100 - min(70, points))
    return score, flags, metrics


def risk_level(score):
    if score >= 85:
        return "Low"
    if score >= 70:
        return "Moderate"
    if score >= 50:
        return "Elevated"
    return "High"


# ---------------- Filing-language review ----------------

RULES = [
    (
        "Accounting / controls",
        "Material weakness language",
        ["material weakness"],
        3,
    ),
    (
        "Accounting / controls",
        "Restatement language",
        ["restatement", "restated financial statements"],
        3,
    ),
    (
        "Liquidity",
        "Going-concern language",
        ["going concern", "substantial doubt about"],
        4,
    ),
    (
        "Liquidity",
        "Debt covenant language",
        ["debt covenant", "covenant violation"],
        2,
    ),
    (
        "Governance",
        "Related-party language",
        ["related party", "related-party"],
        1,
    ),
    (
        "Business quality",
        "Customer concentration language",
        ["customer concentration", "concentration of customers"],
        1,
    ),
    (
        "Business quality",
        "Impairment language",
        ["impairment charge"],
        1,
    ),
    (
        "Business quality",
        "Restructuring language",
        ["restructuring charge"],
        1,
    ),
]


def filing_signals(text):
    lower = text.lower()
    hits = []
    evidence = []

    for category, label, terms, severity in RULES:
        term = next((t for t in terms if t in lower), None)

        if term:
            hits.append(
                {
                    "Category": category,
                    "Signal": label,
                    "Severity": severity,
                    "Trigger": term,
                }
            )
            i = lower.find(term)
            evidence.append(
                (
                    label,
                    text[max(0, i - 260) : i + 520],
                )
            )

    return hits, evidence[:8]


# ---------------- Search / research helpers ----------------

def search_companies(tickers, query, limit=25):
    if not query:
        return tickers.head(limit).copy()

    q = query.strip()
    if not q:
        return tickers.head(limit).copy()

    q_upper = q.upper()

    exact = tickers[tickers["ticker"] == q_upper].copy()

    ticker_match = tickers[
        tickers["ticker"].str.contains(q_upper, na=False)
    ].copy()

    name_match = tickers[
        tickers["name"].str.contains(q, case=False, na=False)
    ].copy()

    combined = pd.concat(
        [exact, ticker_match, name_match],
        ignore_index=True,
    ).drop_duplicates("cik")

    if combined.empty:
        return combined

    combined["_exact"] = (combined["ticker"] == q_upper).astype(int)
    combined["_ticker"] = combined["ticker"].str.startswith(q_upper).astype(int)

    return (
        combined.sort_values(
            ["_exact", "_ticker", "name"],
            ascending=[False, False, True],
        )
        .drop(columns=["_exact", "_ticker"])
        .head(limit)
    )


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

    direction = (
        "positive"
        if value > 0
        else "negative"
        if value < 0
        else "approximately zero"
    )
    return f"{strength} {direction} association"


def score_range_summary(bt):
    frame = bt.copy()
    frame["Score range"] = pd.cut(
        frame["Score"],
        bins=[-1, 69, 84, 100],
        labels=[
            "High risk (<70)",
            "Moderate (70–84)",
            "Low risk (85–100)",
        ],
    )

    summary = (
        frame.groupby(
            "Score range",
            observed=False,
        )["Forward revenue growth"]
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


def historical_observation(ticker, row, facts):
    report_date = row.get("reportDate", "")
    if not report_date:
        return None

    try:
        fy = int(str(report_date)[:4])
    except (TypeError, ValueError):
        return None

    score, flags, metrics = score_year(facts, fy)

    current_rev = value_for_fy(facts, "revenue", fy)
    next_rev = value_for_fy(facts, "revenue", fy + 1)

    return {
        "Ticker": ticker,
        "Fiscal year scored": fy,
        "Filing date": row.get("filingDate", ""),
        "Form": row.get("form", ""),
        "Score": score,
        "Risk level": risk_level(score),
        "Quantitative flags": len(flags),
        "Forward revenue growth": pct_growth(
            current_rev,
            next_rev,
        ),
        "Revenue growth at score date": metrics.get(
            "Revenue growth"
        ),
    }


def run_backtest(tickers_df, years_per_company=5):
    rows = []
    progress = st.progress(
        0,
        text="Building historical observations...",
    )

    for i, ticker in enumerate(SAMPLE):
        match = tickers_df[
            tickers_df["ticker"] == ticker
        ]

        if match.empty:
            continue

        company = match.iloc[0]

        try:
            subs = get_submissions(company["cik"])
            reports = annual_reports(subs)

            if reports.empty:
                continue

            facts = get_companyfacts(company["cik"])
            candidates = []

            for _, row in reports.iterrows():
                report_date = row.get("reportDate")
                if not report_date:
                    continue

                try:
                    fy = int(str(report_date)[:4])
                except (TypeError, ValueError):
                    continue

                if value_for_fy(
                    facts,
                    "revenue",
                    fy + 1,
                ) is not None:
                    candidates.append(row)

            candidates = candidates[-years_per_company:]

            for row in candidates:
                observation = historical_observation(
                    ticker,
                    row,
                    facts,
                )
                if observation:
                    rows.append(observation)

        except Exception:
            # One issuer should never break the entire backtest.
            pass

        progress.progress(
            (i + 1) / len(SAMPLE),
            text=f"Processing {ticker} ({i + 1}/{len(SAMPLE)})...",
        )
        time.sleep(0.05)

    progress.empty()
    return pd.DataFrame(rows)


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
    "SEC EDGAR → XBRL financial data → transparent risk framework → historical test"
)

st.info(
    "Quantitative financial signals and filing-language review signals are kept "
    "separate. A high score means fewer detected quantitative warning conditions; "
    "it does not mean a company is risk-free."
)

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 Company Scanner",
        "📊 Cross-Company Research",
        "🧪 Historical Backtest",
    ]
)

# ============================================================
# COMPANY SCANNER
# ============================================================

with tab1:
    st.subheader("🔎 Company Scanner")
    st.write(
        "Search the SEC-registered company universe by ticker or company name. "
        "The scanner supports annual 10-K, 20-F and 40-F filings."
    )

    try:
        tickers = get_tickers()
    except Exception as exc:
        st.error(
            "The SEC company directory could not be loaded. "
            f"Details: {exc}"
        )
        st.stop()

    query = st.text_input(
        "Search company",
        placeholder="Try Apple, NVIDIA, Tesla, AAPL, NVDA...",
        key="company_search",
    )

    matches = search_companies(tickers, query)

    if query and matches.empty:
        st.warning(
            "No SEC-registered company matched that search. "
            "Try the legal company name or ticker."
        )

    if not matches.empty:
        options = {
            f"{row['ticker']} — {row['name']}": (
                row["ticker"],
                row["cik"],
                row["name"],
            )
            for _, row in matches.iterrows()
        }

        choice = st.selectbox(
            "Select company",
            list(options.keys()),
            key="company_choice",
        )

        ticker, cik, company_name = options[choice]

        st.caption(
            f"SEC CIK: {cik}  •  {len(matches)} matching result(s) shown"
        )

        scan = st.button(
            "Run red-flag scan",
            type="primary",
            key="run_company_scan",
        )

        if scan:
            try:
                with st.spinner(
                    f"Reading the latest annual SEC filing for {company_name}..."
                ):
                    submissions = get_submissions(cik)
                    annual = annual_reports(submissions)

                    if annual.empty:
                        st.error(
                            "No recent annual SEC filing was found. "
                            "This does not necessarily mean the company has no public filing; "
                            "the SEC submission history available to the scanner may not contain "
                            "a supported annual report."
                        )
                        st.stop()

                    latest = annual.iloc[-1]
                    facts = get_companyfacts(cik)

                    try:
                        fy = int(str(latest["reportDate"])[:4])
                    except (TypeError, ValueError):
                        st.error(
                            "The latest annual filing did not contain a usable fiscal year."
                        )
                        st.stop()

                    score, qflags, metrics = score_year(
                        facts,
                        fy,
                    )

                    text, filing_url = get_filing(
                        cik,
                        latest["accessionNumber"],
                        latest["primaryDocument"],
                    )

                    review, evidence = filing_signals(text)

                    st.session_state["company_result"] = {
                        "company": company_name,
                        "ticker": ticker,
                        "cik": cik,
                        "score": score,
                        "level": risk_level(score),
                        "qflags": qflags,
                        "review": review,
                        "evidence": evidence,
                        "metrics": metrics,
                        "filing_date": latest.get(
                            "filingDate",
                            "",
                        ),
                        "report_date": latest.get(
                            "reportDate",
                            "",
                        ),
                        "form": latest.get(
                            "form",
                            "",
                        ),
                        "url": filing_url,
                    }

            except requests.HTTPError as exc:
                st.error(
                    "The SEC returned an HTTP error while retrieving this company. "
                    f"Details: {exc}"
                )
            except requests.RequestException as exc:
                st.error(
                    "The SEC request failed. Check your connection and try again. "
                    f"Details: {exc}"
                )
            except Exception as exc:
                st.error(
                    "The scan could not be completed for this issuer. "
                    f"Details: {exc}"
                )

    if "company_result" in st.session_state:
        result = st.session_state["company_result"]

        st.divider()
        st.subheader(
            f"{result['company']} ({result['ticker']})"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Financial profile score",
            f"{result['score']}/100",
        )
        b.metric(
            "Risk level",
            result["level"],
        )
        c.metric(
            "Quantitative flags",
            len(result["qflags"]),
        )
        d.metric(
            "Filing review signals",
            len(result["review"]),
        )

        st.caption(
            f"Annual filing: {result['form']}  •  "
            f"Filed: {result['filing_date']}  •  "
            f"Fiscal/report year: {result['report_date']}"
        )

        st.caption(
            "The quantitative score excludes filing-language keywords. "
            "Keywords are displayed separately as review signals because their "
            "presence alone does not establish financial distress."
        )

        st.subheader("Quantitative risk signals")

        if result["qflags"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Category": category,
                            "Signal": signal,
                            "Points": points,
                            "Evidence": evidence,
                        }
                        for category, signal, points, evidence
                        in result["qflags"]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success(
                "No predefined quantitative thresholds were triggered."
            )

        st.subheader("Underlying metrics")

        if result["metrics"]:
            metric_rows = []
            for key, value in result["metrics"].items():
                metric_rows.append(
                    {
                        "Metric": key,
                        "Value": (
                            f"{value:.2f}%"
                            if isinstance(value, (int, float))
                            else value
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(metric_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Filing-language review signals")

        if result["review"]:
            st.dataframe(
                pd.DataFrame(result["review"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success(
                "No predefined filing-language review signals were detected."
            )

        st.subheader("Evidence from the filing")

        if result["evidence"]:
            for i, (label, snippet) in enumerate(
                result["evidence"],
                1,
            ):
                with st.expander(
                    f"{i}. {label}"
                ):
                    st.write(snippet)
        else:
            st.caption(
                "No evidence snippets were generated because no predefined "
                "language signal was detected."
            )

        st.subheader("Analyst follow-ups")

        for item in [
            "Reconcile reported earnings with operating/free cash flow.",
            "Investigate working-capital movements relative to revenue.",
            "Inspect debt maturities, covenants and refinancing needs.",
            "Read relevant footnotes and risk factors in context.",
        ]:
            st.write("• " + item)

        st.markdown(
            f"[Open SEC filing]({result['url']})"
        )


# ============================================================
# CROSS-COMPANY RESEARCH
# ============================================================

with tab2:
    st.subheader("📊 Cross-Company Research Study")

    st.write(
        "**Research question:** Can a transparent, automated financial-risk "
        "framework identify meaningful differences in financial profiles "
        "across major public companies?"
    )

    st.info(
        "The sample is intentionally fixed at 20 large public companies so the "
        "methodology is reproducible rather than cherry-picked. This is "
        "descriptive research, not a representative sample of all public companies."
    )

    if st.button(
        "Run 20-company study",
        type="primary",
        key="run_cross_company",
    ):
        try:
            tickers_df = get_tickers()
            rows = []

            progress = st.progress(
                0,
                text="Analyzing companies...",
            )

            for i, ticker in enumerate(SAMPLE):
                match = tickers_df[
                    tickers_df["ticker"] == ticker
                ]

                if match.empty:
                    continue

                company = match.iloc[0]

                try:
                    submissions = get_submissions(
                        company["cik"]
                    )
                    annual = annual_reports(
                        submissions
                    )

                    if annual.empty:
                        continue

                    latest = annual.iloc[-1]
                    facts = get_companyfacts(
                        company["cik"]
                    )

                    fy = int(
                        str(latest["reportDate"])[:4]
                    )

                    score, flags, _ = score_year(
                        facts,
                        fy,
                    )

                    filing_text, _ = get_filing(
                        company["cik"],
                        latest["accessionNumber"],
                        latest["primaryDocument"],
                    )

                    review, _ = filing_signals(
                        filing_text
                    )

                    rows.append(
                        {
                            "Ticker": ticker,
                            "Company": company["name"],
                            "Risk score": score,
                            "Risk level": risk_level(score),
                            "Quantitative flags": len(flags),
                            "Filing review signals": len(review),
                            "Annual form": latest["form"],
                        }
                    )

                except Exception:
                    # Keep the study running if one issuer fails.
                    pass

                progress.progress(
                    (i + 1) / len(SAMPLE),
                    text=f"Processing {ticker} ({i + 1}/{len(SAMPLE)})...",
                )

            progress.empty()

            st.session_state["cross_company"] = pd.DataFrame(
                rows
            )

        except Exception as exc:
            st.error(
                f"The cross-company study failed: {exc}"
            )

    if (
        "cross_company" in st.session_state
        and not st.session_state["cross_company"].empty
    ):
        data = st.session_state["cross_company"].copy()

        data = data.sort_values(
            ["Risk score", "Ticker"],
            ascending=[True, True],
        ).reset_index(drop=True)

        a, b, c = st.columns(3)

        a.metric(
            "Companies analyzed",
            len(data),
        )
        b.metric(
            "Average score",
            f"{data['Risk score'].mean():.1f}/100",
        )
        c.metric(
            "Elevated/high-risk profiles",
            int((data["Risk score"] < 70).sum()),
        )

        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Risk score distribution")

        st.bar_chart(
            data.set_index("Ticker")[["Risk score"]]
        )

        concentration = (
            data["Risk score"].nunique() <= 4
            or (data["Risk score"] >= 85).mean() >= 0.75
        )

        if concentration:
            st.warning(
                "Most companies cluster in the high-score range. "
                "That limited variation makes this cross-sectional comparison "
                "a weak discrimination test; the historical backtest provides "
                "a stronger calibration test."
            )
        else:
            st.info(
                "The sample has meaningful score variation, but the result "
                "remains descriptive and exploratory."
            )

        st.download_button(
            "Download research dataset (CSV)",
            data.to_csv(index=False),
            "sec_red_flag_cross_company_research.csv",
            "text/csv",
            key="download_cross_company",
        )


# ============================================================
# HISTORICAL BACKTEST
# ============================================================

with tab3:
    st.subheader("🧪 Historical Backtest")

    st.write(
        "This test asks a stronger question: when the framework is applied "
        "to an earlier fiscal year, does the resulting score relate to the "
        "company's subsequent reported revenue growth?"
    )

    st.write(
        "The score uses only information available for the selected annual "
        "filing year. The outcome is the following year's reported revenue growth."
    )

    st.warning(
        "This 20-company backtest is exploratory, not statistically conclusive. "
        "Its purpose is to test the framework and generate a falsifiable "
        "research hypothesis. It is not evidence of causation, future stock "
        "returns, or investment performance."
    )

    if st.button(
        "Run historical backtest",
        type="primary",
        key="run_backtest",
    ):
        try:
            tickers_df = get_tickers()
            backtest = run_backtest(
                tickers_df,
                years_per_company=5,
            )

            if backtest.empty:
                st.error(
                    "The backtest returned no usable annual observations."
                )
            else:
                st.session_state["backtest"] = backtest

        except Exception as exc:
            st.error(
                f"The historical backtest failed: {exc}"
            )

    if (
        "backtest" in st.session_state
        and not st.session_state["backtest"].empty
    ):
        bt = st.session_state["backtest"].copy()

        st.metric(
            "Historical observations",
            len(bt),
        )

        st.caption(
            "Annual XBRL duration filters are applied to revenue, net income "
            "and operating cash flow to reduce distorted period selections."
        )

        st.dataframe(
            bt,
            use_container_width=True,
            hide_index=True,
        )

        clean = bt.dropna(
            subset=[
                "Score",
                "Forward revenue growth",
            ]
        ).copy()

        if len(clean) >= 3:
            pearson = clean["Score"].corr(
                clean["Forward revenue growth"],
                method="pearson",
            )

            # Rank-based Spearman calculation without scipy.
            score_rank = clean["Score"].rank(
                method="average"
            )
            growth_rank = clean[
                "Forward revenue growth"
            ].rank(method="average")

            spearman = score_rank.corr(
                growth_rank
            )

            a, b, c = st.columns(3)

            a.metric(
                "Pearson correlation",
                (
                    "Not available"
                    if pd.isna(pearson)
                    else f"{pearson:.2f}"
                ),
            )

            b.metric(
                "Spearman correlation",
                (
                    "Not available"
                    if pd.isna(spearman)
                    else f"{spearman:.2f}"
                ),
            )

            c.metric(
                "Unique score values",
                int(clean["Score"].nunique()),
            )

            st.subheader(
                "Framework score vs. subsequent revenue growth"
            )

            st.scatter_chart(
                clean,
                x="Score",
                y="Forward revenue growth",
                x_label="Historical framework score",
                y_label="Following-year revenue growth (%)",
            )

            if not pd.isna(pearson):
                st.info(
                    f"In this exploratory sample, the Pearson correlation is "
                    f"**{pearson:.2f}**, indicating a **{correlation_label(pearson)}** "
                    "between framework score and following-year revenue growth. "
                    "This is an association, not evidence of causation or "
                    "investment performance."
                )

            if not pd.isna(spearman):
                st.caption(
                    f"Spearman ρ = {spearman:.2f}. This rank-based result is "
                    "included as a robustness check."
                )

            summary = score_range_summary(clean)

            st.subheader(
                "Subsequent growth by score range"
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

            counts = summary[
                "Observations"
            ].fillna(0)

            if len(clean) and counts.max() > 0.75 * len(clean):
                st.warning(
                    "Sample-balance limitation: most observations fall in "
                    "one score range. Comparisons between ranges should "
                    "therefore be treated cautiously."
                )

            st.caption(
                "Score-range results are descriptive. The backtest tests "
                "association with subsequent reported revenue growth, not "
                "stock returns, valuation, fraud, or causation."
            )

        st.download_button(
            "Download historical backtest (CSV)",
            bt.to_csv(index=False),
            "sec_red_flag_historical_backtest.csv",
            "text/csv",
            key="download_backtest",
        )


# ============================================================
# METHODOLOGY
# ============================================================

st.divider()
st.subheader("Methodology")

st.markdown(
    """
**Quantitative scoring**

The score starts at 100 and subtracts points only when predefined
quantitative warning conditions are triggered:

- Revenue decline ≥ 5%: 6 points
- Receivables growth ≥ 10 percentage points above revenue growth: 10 points
- Negative operating cash flow despite positive net income: 12 points
- Operating cash flow below 80% of positive net income: 10 points
- Debt growth ≥ 10%: 8 points
- Inventory growth ≥ 15%: 6 points
- Net income decline ≥ 10%: 6 points

The score is capped at a 70-point maximum deduction.

A score of 100 therefore means that none of the predefined quantitative
conditions was detected in the available XBRL data. It does **not** mean
the company is safe, high quality, or free of accounting risk.

**Filing-language review**

Material weakness, restatement, going-concern, covenant, related-party,
customer-concentration, impairment and restructuring terms are displayed
as review signals rather than being mixed into the quantitative score.
Keyword presence alone does not establish financial distress.

**Company coverage**

The search is based on the SEC's company directory and supports SEC-registered
issuers whose annual filings are available as 10-K, 20-F or 40-F reports.
This is broader than a U.S.-only 10-K search, but it is not literally every
company listed on every stock exchange worldwide.

**Historical test**

For each sample company, earlier annual filings are selected only when the
following year's revenue is available. The score is calculated using the
earlier fiscal year and compared with subsequent reported revenue growth.

Pearson correlation measures linear association. Spearman correlation is
calculated from ranks and does not require scipy.

The backtest is exploratory and is not evidence of causation, fraud detection,
future stock returns, or investment performance.
"""
)
