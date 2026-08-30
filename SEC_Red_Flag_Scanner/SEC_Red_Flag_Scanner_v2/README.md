# SEC Filing Red-Flag Scanner — v2

A finance-research prototype combining SEC filing-language analysis with SEC XBRL financial-statement data.

## What changed from v1

The original version treated generic words such as "liquidity" or "accounts receivable" as red flags. v2 adds quantitative tests and a capped scoring framework.

### Quantitative tests
- Receivables growth materially above revenue growth
- Operating cash flow materially below net income
- Debt growth
- Inventory growth

### Filing-language tests
- Material weaknesses / restatements
- Going-concern language
- Related parties
- Customer concentration
- Debt covenant language
- Liquidity constraints
- Restructuring
- Impairments
- Negative guidance language

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Before deployment, replace the placeholder SEC User-Agent contact email with your own.

## Research extension

Run the framework across a fixed sample of companies and compare risk scores with subsequent financial performance. Preserve the raw scores and methodology so the study is reproducible.

## Resume framing

Built an SEC filing due-diligence tool combining filing-language analysis with XBRL financial data to flag accounting, liquidity, working-capital and business-quality risks; developed a transparent scoring framework and evidence-backed analyst reports.

Educational/research prototype only; not investment advice.
