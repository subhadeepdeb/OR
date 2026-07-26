"""
Solve the enhanced airline crew scheduling MILP using scipy.optimize.milp.

The formulation is a set-partitioning master problem with additional operational
constraints for base/qualification/day capacity, reserves, hotels, deadhead seats,
workload balance, and stochastic disruption exposure.

Run:
    python src/solve_crew_scheduling_milp.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, vstack

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

flights = pd.read_csv(RAW / "flights.csv")
airports = pd.read_csv(RAW / "airports.csv")
availability = pd.read_csv(RAW / "crew_availability.csv")
pairings = pd.read_csv(PROC / "candidate_pairings.csv")
legs = pd.read_csv(PROC / "pairing_legs.csv")
scenarios = pd.read_csv(RAW / "delay_scenarios.csv")
scenario_probs = scenarios.groupby("scenario")["probability"].first().to_dict()
scenario_names = list(scenario_probs)

nP = len(pairings)
reserve_keys = [(int(r.day), r.base, r.qualification_pool) for r in availability.itertuples(index=False)]
nR = len(reserve_keys)
base_keys = sorted(pairings.base.unique())
nB = len(base_keys)
nS = len(scenario_names)

# Variable blocks: x[p] binary; reserve[r] integer; dev_plus/base and dev_minus/base continuous; risk_slack[s] continuous.
offset_x = 0
offset_r = nP
offset_dp = nP + nR
offset_dm = offset_dp + nB
offset_risk_slack = offset_dm + nB
N = offset_risk_slack + nS

c = np.zeros(N)
c[offset_x:offset_x+nP] = pairings.cost.values
c[offset_r:offset_r+nR] = 420.0  # reserve crew opportunity cost
c[offset_dp:offset_dp+nB] = 65.0 # fairness/workload imbalance penalty
c[offset_dm:offset_dm+nB] = 65.0
c[offset_risk_slack:offset_risk_slack+nS] = 900.0 # penalty if scenario risk budget exceeded

integrality = np.zeros(N, dtype=int)
integrality[offset_x:offset_x+nP] = 1
integrality[offset_r:offset_r+nR] = 1

lb = np.zeros(N)
ub = np.full(N, np.inf)
ub[offset_x:offset_x+nP] = 1
ub[offset_r:offset_r+nR] = 10
bounds = Bounds(lb, ub)

constraints = []
lowers = []
uppers = []
row_names = []

def add_row(coeffs: dict[int, float], lo: float, hi: float, name: str):
    row = lil_matrix((1, N))
    for j, v in coeffs.items():
        if abs(v) > 1e-9:
            row[0, j] = v
    constraints.append(row.tocsr())
    lowers.append(lo)
    uppers.append(hi)
    row_names.append(name)

# 1) Set partitioning: every flight covered exactly once.
for f in flights.flight_id:
    pids = legs.loc[legs.flight_id == f, "pairing_id"]
    idx = pairings.index[pairings.pairing_id.isin(pids)].tolist()
    add_row({offset_x+i: 1 for i in idx}, 1, 1, f"cover_{f}")

# 2) Crew availability by day/base/qualification pool, reserves consume capacity too.
for ridx, (day, base, qual) in enumerate(reserve_keys):
    avail = int(availability[(availability.day == day) & (availability.base == base) & (availability.qualification_pool == qual)].available_crews.iloc[0])
    min_res = int(availability[(availability.day == day) & (availability.base == base) & (availability.qualification_pool == qual)].minimum_reserve.iloc[0])
    # selected pairings that start this day/base and can use this pool. BOTH pairings can draw from either NB/RJ pool; keep conservative by direct pool match + BOTH.
    idx = pairings.index[(pairings.start_day == day) & (pairings.base == base) & (pairings.qualification_pool.isin([qual, "BOTH"]))].tolist()
    coeffs = {offset_x+i: 1 for i in idx}
    coeffs[offset_r+ridx] = 1
    add_row(coeffs, -np.inf, avail, f"crew_avail_D{day}_{base}_{qual}")
    add_row({offset_r+ridx: 1}, min_res, np.inf, f"min_reserve_D{day}_{base}_{qual}")

# 3) Hotel room capacity for overnight pairings by station-night.
hotel_rows = pairings[(pairings.hotel_nights > 0) & (pairings.overnight_station.notna()) & (pairings.overnight_station != "")]
for (station, day), group in hotel_rows.groupby(["overnight_station", "hotel_day"]):
    cap = int(airports.loc[airports.airport == station, "hotel_rooms_per_night"].iloc[0])
    add_row({offset_x+i: 1 for i in group.index}, -np.inf, cap, f"hotel_{station}_D{int(day)}")

# 4) Deadhead seat budget by base and start day. This avoids unrealistic repositioning.
for (base, day), group in pairings.groupby(["base", "start_day"]):
    coeffs = {offset_x+i: float(pairings.loc[i, "deadhead_min"] > 0) for i in group.index}
    # cap roughly allows some repositioning but discourages solving all flights via remote bases.
    add_row(coeffs, -np.inf, 10, f"deadhead_budget_{base}_D{int(day)}")

# 5) Workload balance by base over the horizon: duty minutes should stay near target.
total_block = float(flights.block_min.sum())
target_duty = float(pairings.duty_min.mean() * (len(flights) / max(pairings.num_legs.mean(), 1)) / len(base_keys))
# Use a practical target by planned base share: 25 selected pairings avg 250 duty / 4-ish.
target_duty = 0.25 * pairings.duty_min.median() * len(flights) / 1.8
for bidx, base in enumerate(base_keys):
    idx = pairings.index[pairings.base == base].tolist()
    coeffs1 = {offset_x+i: pairings.loc[i, "duty_min"] for i in idx}
    coeffs1[offset_dp+bidx] = -1
    add_row(coeffs1, -np.inf, target_duty, f"workload_upper_{base}")
    coeffs2 = {offset_x+i: -pairings.loc[i, "duty_min"] for i in idx}
    coeffs2[offset_dm+bidx] = -1
    add_row(coeffs2, -np.inf, -0.65*target_duty, f"workload_lower_{base}")

# 6) Stochastic disruption exposure budget: expected misconnect/overtime risk per scenario.
for sidx, s in enumerate(scenario_names):
    coeffs = {offset_x+i: float(pairings.loc[i, f"risk_{s}"]) for i in range(nP)}
    coeffs[offset_risk_slack+sidx] = -1
    # weather scenarios can tolerate more risk, but slack is expensive.
    budget = 900 if s == "S0_nominal" else 1450
    add_row(coeffs, -np.inf, budget, f"risk_budget_{s}")

A = vstack(constraints).tocsr()
lin = LinearConstraint(A, np.array(lowers), np.array(uppers))

result = milp(c=c, integrality=integrality, bounds=bounds, constraints=lin, options={"time_limit": 90, "mip_rel_gap": 0.01, "disp": False})

if not result.success and result.x is None:
    raise RuntimeError(f"MILP did not return a feasible solution: {result.message}")

x = result.x[offset_x:offset_x+nP]
selected_idx = np.where(x > 0.5)[0]
selected = pairings.iloc[selected_idx].copy().sort_values(["start_day", "base", "cost"])
selected_legs = legs[legs.pairing_id.isin(selected.pairing_id)].merge(flights, on="flight_id", how="left")
reserve_sol = pd.DataFrame(reserve_keys, columns=["day","base","qualification_pool"])
reserve_sol["reserve_crews"] = np.rint(result.x[offset_r:offset_r+nR]).astype(int)
base_workload = selected.groupby("base").agg(
    selected_pairings=("pairing_id","count"), duty_min=("duty_min","sum"), block_min=("block_min","sum"),
    deadhead_min=("deadhead_min","sum"), avg_expected_delay_risk=("expected_delay_risk","mean")
).reset_index()
base_workload["duty_hours"] = (base_workload.duty_min / 60).round(2)

coverage = selected_legs.groupby("flight_id").pairing_id.nunique().reset_index(name="times_covered")
coverage_check = flights[["flight_id"]].merge(coverage, on="flight_id", how="left").fillna({"times_covered":0})

risk_summary = []
for sidx, s in enumerate(scenario_names):
    risk = float(sum(pairings.loc[i, f"risk_{s}"] for i in selected_idx))
    slack = float(result.x[offset_risk_slack+sidx])
    risk_summary.append({"scenario":s,"probability":scenario_probs[s],"selected_pairing_risk_min":round(risk,2),"risk_slack_min":round(slack,2)})
risk_summary = pd.DataFrame(risk_summary)

summary = pd.DataFrame([
    {"metric":"solver_status", "value":str(result.message)},
    {"metric":"objective_value", "value":round(float(result.fun),2)},
    {"metric":"candidate_pairings", "value":nP},
    {"metric":"selected_pairings", "value":len(selected)},
    {"metric":"flights", "value":len(flights)},
    {"metric":"all_flights_covered_exactly_once", "value":bool((coverage_check.times_covered == 1).all())},
    {"metric":"recovery_single_pairings_selected", "value":int((selected.pairing_type == "recovery_single").sum())},
    {"metric":"round_trip_pairings_selected", "value":int((selected.pairing_type == "round_trip").sum())},
    {"metric":"overnight_pairings_selected", "value":int((selected.pairing_type == "overnight").sum())},
    {"metric":"three_leg_pairings_selected", "value":int((selected.pairing_type == "three_leg").sum())},
    {"metric":"total_reserve_crews", "value":int(reserve_sol.reserve_crews.sum())},
    {"metric":"max_base_duty_hours", "value":round(float(base_workload.duty_hours.max()),2)},
    {"metric":"min_base_duty_hours", "value":round(float(base_workload.duty_hours.min()),2)},
    {"metric":"total_deadhead_hours", "value":round(float(selected.deadhead_min.sum()/60),2)},
    {"metric":"total_expected_delay_risk_min", "value":round(float(selected.expected_delay_risk.sum()),2)},
    {"metric":"mip_gap", "value":getattr(result, 'mip_gap', None)},
])

selected.to_csv(OUT / "selected_pairings.csv", index=False)
selected_legs.to_csv(OUT / "selected_pairing_legs.csv", index=False)
reserve_sol.to_csv(OUT / "reserve_plan.csv", index=False)
base_workload.to_csv(OUT / "base_workload_summary.csv", index=False)
coverage_check.to_csv(OUT / "flight_coverage_check.csv", index=False)
risk_summary.to_csv(OUT / "scenario_risk_summary.csv", index=False)
summary.to_csv(OUT / "solution_summary.csv", index=False)

print(summary.to_string(index=False))
