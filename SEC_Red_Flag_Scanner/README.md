# SEC Filing Risk Lab v7.2

An educational financial-research prototype built with Python, Streamlit, SEC EDGAR and XBRL data. It screens public-company filings for predefined financial warning signals, surfaces filing evidence for review, and tests whether those signals are associated with subsequent accounting and stock-price outcomes.

## Features

### Company Scanner
- Retrieve company information and standard 10-K filings from SEC EDGAR.
- Calculate a transparent, rule-based financial profile score.
- Display quantitative warning signals separately from filing-language review prompts.
- Show supporting financial metrics, filing evidence and analyst follow-up questions.

### Quantitative Financial Signals
The current scoring framework tests four predefined accounting patterns:

- **Receivables:** Receivables growth exceeds revenue growth by more than 20 percentage points.
- **Cash conversion:** Operating cash flow is below 60% of positive net income.
- **Leverage:** Debt growth is at least 20%.
- **Inventory:** Inventory growth is at least 25%.

The score is a screening heuristic, not a probability of default, a fraud determination or an investment recommendation. Missing observations are not automatically treated as negative signals.

### Historical Research and Backtesting
- Compare individual financial flags with subsequent fiscal-year revenue growth.
- Report flagged-versus-unflagged outcomes, means, medians, Pearson correlations and Spearman rank correlations.
- Display uncertainty estimates for median differences when the available sample supports them.
- Compare point-in-time results with a deliberately unrestricted look-ahead-bias diagnostic. The unrestricted result is not a valid historical backtest.
- Choose the 20-company pilot, 100-company research sample, fixed 10-company holdout or a custom sample.
- Export historical observations and research tables to CSV.

### Stock-Price Drawdown Analysis
When enabled, the application retrieves adjusted historical stock prices and calculates maximum drawdown over the next 252 trading-day closes following a filing.

It also compares the prevalence of individual warning signals in the worst drawdown quartile.

This feature requires network access and the `yfinance` dependency. Incomplete price histories are reported as unavailable rather than treated as complete results.

### Filing-Language Review
- Search 10-K filing text for predefined accounting, liquidity, governance and business-quality review signals.
- Display supporting excerpts for manual examination.
- Compare Risk Factors and Management's Discussion and Analysis (MD&A) sections across consecutive annual filings using TF-IDF and cosine similarity.
- Surface changes in disclosure language and terms for further investigation.

Text extraction is heuristic. Similarity scores and keyword matches do not establish management intent, misconduct or financial distress.

### XBRL and Financial Context
- Prefer standard US-GAAP concepts where suitable observations are available.
- Inspect eligible company-specific XBRL extension concepts as potential fallbacks when the relevant research option is enabled.
- Exclude ambiguous or competing mappings rather than automatically forcing them into the dataset.
- Search the first two subsequent 10-Q filings for relevant passages about receivables, collections, payment terms and cash conversion.
- Calculate a descriptive accruals proxy:

  `(Net income − Operating cash flow) / Average total assets × 100`

The accruals proxy is not a complete accrual-accounting decomposition and does not receive a separate scoring penalty.

## Installation and Setup

### Requirements
- Python 3.10 or later
- Internet access for SEC filings and optional historical stock-price retrieval

### Run on macOS or Linux

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Configure a meaningful SEC User-Agent with a real contact email:

```bash
export SEC_USER_AGENT="SEC Filing Risk Lab/7.2 your-real-contact-email@example.com"
```

Replace the placeholder email before using SEC endpoints at scale. Follow the SEC's fair-access guidance.

Start the application:

```bash
streamlit run app.py
```

### Run on Windows

Create the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SEC_USER_AGENT="SEC Filing Risk Lab/7.2 your-real-contact-email@example.com"
streamlit run app.py
```

## Research Methodology

Historical financial signals are evaluated using information constrained to the relevant filing date wherever the point-in-time research workflow applies. Forward revenue growth is measured as:

`(Following-year revenue / Current-year revenue − 1) × 100`

Maximum drawdown measures the largest peak-to-trough decline within the specified forward stock-price window.

The application separates the observations used to construct a warning signal from the outcomes used to evaluate it. Association does not establish causation or prove that a signal provides profitable investment decisions.

## Limitations

1. The 20-company pilot contains established listed companies selected from today's market. Survivorship and sample-selection bias may affect the findings.
2. A small sample can produce unstable correlations and misleading apparent patterns. Larger samples do not eliminate all sources of bias.
3. Companies differ in accounting concepts, reporting periods, fiscal calendars and disclosure structure. XBRL mappings should be checked against source filings.
4. Risk Factors and MD&A extraction may fail on unusual filing layouts or formatting. Review extracted text before interpreting similarity results.
5. Searches of subsequent 10-Q filings do not prove that a particular proportion of receivables was collected unless the disclosed data support that calculation.
6. The fixed 10-company holdout is a new, separately specified sample. It should not be presented as the exact sample used in an earlier experiment unless the original company list matches.
7. Results described in application essays—including particular correlations, drawdown patterns, flagged-company counts and company-specific examples—must be independently reproduced from the relevant data. The software does not guarantee those findings.
8. Repeated company-year observations are not independent. Research-grade analysis should account for within-company dependence, missing data, industry differences and sample selection.

## Data Sources and Dependencies

- SEC EDGAR company directory, filing submissions, XBRL Company Facts and filing documents.
- Historical adjusted stock-price data through `yfinance`, when price analysis is enabled.
- Python packages listed in `requirements.txt`, including Streamlit, NumPy, pandas, requests, Beautiful Soup and scikit-learn.

## Disclaimer

This project is an educational research prototype. Its signals are prompts for further investigation, not investment advice or verified evidence of fraud, default or future market performance.
