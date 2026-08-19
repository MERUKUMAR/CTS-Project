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

This opens the dashboard in your browser at `http://localhost:8501`.
**This works immediately** using the bundled sample data (29 ACOs) — no internet
or extra setup needed.

---

## (Optional) Get the FULL dataset (~700+ ACOs)

The app ships with a 29-ACO sample so it always works out of the box. To pull the
complete live dataset from CMS instead:

```
python scripts/etl.py
```

This fetches from `data.cms.gov`, cleans it, and saves:
- `data/aco_scorecard_full.csv`

The app automatically prefers this file over the sample if it exists — just
re-run `streamlit run app.py` after generating it (or refresh the browser tab).

---

## What each page does

| Page | Purpose |
|---|---|
| **Portfolio Overview** | Sortable/filterable table of every ACO — find where to focus |
| **Contract Scorecard** | Drill into one ACO: cost vs benchmark, quality, utilization |
| **Driver Analysis** | The "why" — which cost category is driving variance vs peers |
| **Recommendations** | Rule-based action list + downloadable meeting brief (.md) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'X'`**
Your venv isn't activated, or `pip install -r requirements.txt` didn't run
inside it. Check your prompt shows `(venv)`, then re-run the install command.

**`streamlit: command not found`**
Same cause — activate the venv first.

**Port already in use**
Run `streamlit run app.py --server.port 8502` instead.
