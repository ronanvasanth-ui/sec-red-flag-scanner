import re
import pandas as pd
import requests
import streamlit as st

SEC_HEADERS={"User-Agent":"SEC Filing Red-Flag Scanner/1.0 contact@example.com"}

@st.cache_data(ttl=3600)
def get_tickers():
    r=requests.get("https://www.sec.gov/files/company_tickers.json",headers=SEC_HEADERS,timeout=20)
    r.raise_for_status()
    return pd.DataFrame([{"ticker":v["ticker"],"name":v["title"],"cik":str(v["cik_str"]).zfill(10)} for v in r.json().values()])

@st.cache_data(ttl=3600)
def get_submissions(cik):
    r=requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",headers=SEC_HEADERS,timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def get_filing(cik, accession, document):
    url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-','')}/{document}"
    r=requests.get(url,headers=SEC_HEADERS,timeout=30); r.raise_for_status()
    return re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",r.text)),url

def analyze(text):
    t=text.lower(); flags=[]
    rules=[
      ("Going-concern language",["going concern","substantial doubt"],3),
      ("Material weakness",["material weakness"],3),
      ("Restatement/accounting correction",["restatement","restated financial statements"],3),
      ("Debt/covenant pressure",["debt covenant","covenant violation","covenant"],1),
      ("Liquidity pressure",["liquidity risk","liquidity constraints","liquidity"],1),
      ("Asset impairment",["impairment charge","impairment"],2),
      ("Related-party activity",["related party","related-party"],2),
      ("Customer concentration",["customer concentration","concentration of customers"],2),
      ("Restructuring",["restructuring charge","restructuring"],1),
      ("Negative outlook/guidance",["lowered our guidance","lower our guidance","below our previous guidance","headwinds"],1),
      ("Revenue/receivables review",["accounts receivable","trade receivables"],1),
      ("Cash-flow quality review",["operating cash flow","cash provided by operating activities"],1)
    ]
    snippets=[]
    for label,terms,w in rules:
        hit=next((x for x in terms if x in t),None)
        if hit:
            flags.append((label,w,hit))
            i=t.find(hit); snippets.append(text[max(0,i-220):i+420])
    score=max(0,100-sum(w*8 for _,w,_ in flags))
    return score,flags,snippets[:8]

def report(company,date,score,flags,snips):
    level="LOW" if score>=80 else "MEDIUM" if score>=60 else "HIGH"
    s=f"# SEC Filing Red-Flag Report — {company}\n**Filing date:** {date}\n**Score:** {score}/100 ({level} risk)\n\n## Signals\n"
    s+="\n".join(f"- **{a}** — trigger: `{c}`" for a,_,c in flags) or "No major heuristic signals detected."
    s+="\n\n## Evidence\n"+("\n\n".join("> "+x for x in snips) if snips else "No snippets captured.")
    s+="\n\n## Analyst follow-ups\n- Reconcile earnings with operating/free cash flow.\n- Inspect debt maturities and covenants.\n- Compare receivables/working capital with revenue.\n- Read the relevant footnotes and risk factors.\n\n*Educational prototype; not investment advice.*"
    return s

st.set_page_config(page_title="SEC Red-Flag Scanner",page_icon="🔎",layout="wide")
st.title("🔎 SEC Filing Red-Flag Scanner")
st.caption("SEC EDGAR → 10-K → transparent financial-risk heuristics → evidence → analyst follow-ups")

try: tickers=get_tickers()
except Exception as e: st.error(f"SEC company list unavailable: {e}"); st.stop()

q=st.text_input("Search company",placeholder="Apple, NVIDIA, Tesla...")
m=tickers if not q else tickers[tickers.ticker.str.contains(q.upper(),na=False)|tickers.name.str.contains(q,case=False,na=False)]
m=m.head(25)
opts={f"{r.ticker} — {r.name}":(r.ticker,r.cik,r.name) for _,r in m.iterrows()}
choice=st.selectbox("Company",list(opts) or ["No matches"])

if st.button("Run red-flag scan",type="primary"):
    ticker,cik,company=opts[choice]
    try:
        rec=get_submissions(cik)["filings"]["recent"]; df=pd.DataFrame(rec)
        df=df[df.form=="10-K"].sort_values("filingDate",ascending=False)
        row=df.iloc[0]
        text,url=get_filing(cik,row.accessionNumber,row.primaryDocument)
        score,flags,snips=analyze(text)
        st.session_state.update(report=report(company,row.filingDate,score,flags,snips),score=score,flags=flags,snips=snips,url=url,company=company)
    except Exception as e: st.error(f"SEC retrieval failed: {e}"); st.stop()

if "score" in st.session_state:
    a,b,c=st.columns(3)
    a.metric("Red-flag score",f"{st.session_state.score}/100")
    b.metric("Signals",len(st.session_state.flags))
    c.metric("Evidence snippets",len(st.session_state.snips))
    st.subheader("Signals")
    st.dataframe(pd.DataFrame([{"Signal":x[0],"Severity":x[1],"Trigger":x[2]} for x in st.session_state.flags]),use_container_width=True,hide_index=True)
    st.subheader("Evidence")
    for i,x in enumerate(st.session_state.snips,1):
        with st.expander(f"Evidence {i}"): st.write(x)
    st.subheader("Analyst follow-ups")
    for x in ["Reconcile earnings with operating/free cash flow.","Inspect debt maturities and covenants.","Compare receivables/working capital with revenue.","Read relevant footnotes and risk factors."]: st.write("• "+x)
    st.download_button("Download analyst report",st.session_state.report,file_name=f"{st.session_state.company.replace(' ','_')}_red_flags.md",mime="text/markdown")
    st.markdown(f"[Open SEC filing]({st.session_state.url})")
