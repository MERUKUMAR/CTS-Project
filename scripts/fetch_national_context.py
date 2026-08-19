"""Pulls NATIONAL market-context benchmarks (not per-ACO joined)."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "national_context.json"

HOSPITAL_VBP_DATASET_ID = "ypbt-wvdk"
DATASTORE_QUERY = "https://data.cms.gov/provider-data/api/1/datastore/query/{id}/0"
PHYSICIAN_GEO_SEARCH_URL = "https://data.cms.gov/data.json"


def fetch_hospital_vbp_avg() -> dict | None:
    try:
        resp = requests.get(DATASTORE_QUERY.format(id=HOSPITAL_VBP_DATASET_ID),
                             params={"limit": 5000}, timeout=60)
        resp.raise_for_status()
        rows = resp.json().get("results", resp.json()) if resp.content else []
        if not rows:
            return None
        scores = []
        for r in rows:
            for key in ("Total Performance Score", "total_performance_score", "TPS"):
                if key in r and r[key] not in (None, "", "Not Available"):
                    try:
                        scores.append(float(r[key]))
                    except (ValueError, TypeError):
                        pass
                    break
        if not scores:
            return None
        return {"avg_total_performance_score": round(sum(scores) / len(scores), 1),
                "n_hospitals": len(scores), "fetched": date.today().isoformat()}
    except Exception as e:
        print(f"  Hospital VBP fetch failed: {e}")
        return None


def fetch_physician_geo_national_avg() -> dict | None:
    try:
        catalog = requests.get(PHYSICIAN_GEO_SEARCH_URL, timeout=60).json()
        dataset_meta = next(
            (d for d in catalog.get("dataset", [])
             if "Physician & Other Practitioners" in d.get("title", "")
             and "Geography" in d.get("title", "")), None)
        if not dataset_meta:
            return None
        csv_url = next((d["downloadURL"] for d in dataset_meta.get("distribution", [])
                         if d.get("mediaType") == "text/csv"), None)
        if not csv_url:
            return None
        import pandas as pd
        chunks = pd.read_csv(csv_url, usecols=lambda c: c in (
            "Rndrng_Prvdr_Geo_Lvl", "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Pymt_Amt", "HCPCS_Cd"
        ), chunksize=200_000, low_memory=False)
        payments = []
        for chunk in chunks:
            nat = chunk[chunk["Rndrng_Prvdr_Geo_Lvl"] == "National"]
            if not nat.empty and "Avg_Mdcr_Pymt_Amt" in nat.columns:
                payments.extend(nat["Avg_Mdcr_Pymt_Amt"].dropna().tolist())
        if not payments:
            return None
        return {"avg_medicare_payment_amt": round(sum(payments) / len(payments), 2),
                "n_services_rows": len(payments), "fetched": date.today().isoformat()}
    except Exception as e:
        print(f"  Physician & Other Practitioners fetch failed: {e}")
        return None


def run():
    print("Fetching national market-context benchmarks...")
    context = {}
    hvbp = fetch_hospital_vbp_avg()
    if hvbp:
        context["hospital_vbp"] = hvbp
    phys = fetch_physician_geo_national_avg()
    if phys:
        context["physician_geo"] = phys
    if not context:
        print("No national context could be fetched — dashboard will skip the panel.")
        return
    with open(OUT_PATH, "w") as f:
        json.dump(context, f, indent=2)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    run()