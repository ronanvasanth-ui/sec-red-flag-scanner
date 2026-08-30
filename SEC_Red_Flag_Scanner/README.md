# SEC Filing Red-Flag Scanner

One-day finance/AI portfolio MVP.

### What it does
Search a public company, pull its latest 10-K from SEC EDGAR, scan for transparent warning signals, show evidence snippets, and export an analyst-style report.

### Signals
Going concern, material weaknesses, restatements, debt/covenants, liquidity, impairments, related parties, customer concentration, restructuring, negative guidance, receivables/revenue review, and cash-flow quality.

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Replace the placeholder SEC User-Agent contact email in `app.py` before public deployment.

### Stronger version for admissions/recruiting
Run it over 30–100 companies and test whether higher red-flag scores correlate with weaker subsequent free-cash-flow growth or stock performance. Include 3–5 case studies and false-positive examples.

### Resume bullet
Built an SEC filing red-flag scanner that parses 10-Ks and flags potential liquidity, accounting, governance, and revenue-quality risks; generated evidence-backed analyst reports and diligence questions.

Educational research prototype only; not investment advice.
