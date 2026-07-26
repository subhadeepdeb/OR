from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

sheets = {
    "solution_summary": OUT / "solution_summary.csv",
    "selected_pairings": OUT / "selected_pairings.csv",
    "selected_pairing_legs": OUT / "selected_pairing_legs.csv",
    "reserve_plan": OUT / "reserve_plan.csv",
    "base_workload_summary": OUT / "base_workload_summary.csv",
    "scenario_risk_summary": OUT / "scenario_risk_summary.csv",
    "flight_coverage_check": OUT / "flight_coverage_check.csv",
    "candidate_pairings": PROC / "candidate_pairings.csv",
    "flights": RAW / "flights.csv",
    "airports": RAW / "airports.csv",
    "crew_availability": RAW / "crew_availability.csv",
}

xlsx = OUT / "airline_crew_scheduling_outputs.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    for name, path in sheets.items():
        if path.exists():
            df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=name[:31], index=False)
print(f"Wrote {xlsx}")
