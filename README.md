# Value-Based Care Command Center

A payer-facing dashboard that tracks provider/ACO performance on **cost, quality,
and utilization**, identifies what's driving savings or losses, and recommends
actions for the provider review meeting.

Built on real CMS Medicare Shared Savings Program (MSSP) data.

---

## Folder structure

```
vbc_dashboard/
├── app.py                        <- Streamlit entry point (run this)
├── data_utils.py                 <- shared data-loading helpers
├── requirements.txt
├── data/
│   ├── raw_aco_data_sample.json  <- 29 real ACOs, bundled so it works offline
│   └── aco_scorecard.csv         <- cleaned + enriched sample (pre-built)
├── scripts/
│   └── etl.py                    <- fetches ALL ~700+ ACOs from CMS (needs internet)
└── pages/
    ├── 1_Portfolio_Overview.py
    ├── 2_Contract_Scorecard.py
    ├── 3_Driver_Analysis.py
    └── 4_Recommendations.py
```

---

## Setup (Windows)

Open Command Prompt or PowerShell, `cd` into this folder, then:

```bat
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

If you're using PowerShell instead of Command Prompt and activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
venv\Scripts\Activate.ps1
```

You'll know the venv is active when your prompt shows `(venv)` at the start.

## Setup (Mac/Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Run the app

With the venv active:

```
streamlit run app.py
```
