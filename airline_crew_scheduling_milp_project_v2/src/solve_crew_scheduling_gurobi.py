from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise ImportError(
        "gurobipy is not installed. Install it with: pip install gurobipy\n"
        "You will also need a valid Gurobi license."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all raw and processed project files needed by the MILP."""
    flights = pd.read_csv(RAW / "flights.csv")
    airports = pd.read_csv(RAW / "airports.csv")
    availability = pd.read_csv(RAW / "crew_availability.csv")
    pairings = pd.read_csv(PROC / "candidate_pairings.csv")
    legs = pd.read_csv(PROC / "pairing_legs.csv")
    scenarios = pd.read_csv(RAW / "delay_scenarios.csv")

    # Make string columns safe for grouping/comparisons.
    for df in [flights, airports, availability, pairings, legs, scenarios]:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("")

    return flights, airports, availability, pairings, legs, scenarios


def solve_with_gurobi(
    time_limit: int = 120,
    mip_gap: float = 0.01,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
   
    flights, airports, availability, pairings, legs, scenarios = load_inputs()

    scenario_probs = scenarios.groupby("scenario")["probability"].first().to_dict()
    scenario_names = list(scenario_probs)
    base_keys = sorted(pairings.base.unique())
    reserve_keys = [
        (int(row.day), row.base, row.qualification_pool)
        for row in availability.itertuples(index=False)
    ]

    # Helpful index maps for fast constraint construction.
    pairing_ids = pairings["pairing_id"].tolist()
    pairings_by_id = pairings.set_index("pairing_id", drop=False)
    legs_by_flight = legs.groupby("flight_id")["pairing_id"].apply(list).to_dict()

    m = gp.Model("enhanced_airline_crew_scheduling_set_partitioning")
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.OutputFlag = 1 if verbose else 0

    # Optional: improve numerical behavior and MIP search robustness.
    m.Params.NumericFocus = 1
    m.Params.Heuristics = 0.15
    m.Params.Cuts = 2

    # -------------------------
    # Decision variables
    # -------------------------
    x = m.addVars(pairing_ids, vtype=GRB.BINARY, name="select_pairing")

    reserve = m.addVars(
        range(len(reserve_keys)),
        vtype=GRB.INTEGER,
        lb=0,
        ub=10,
        name="reserve_crews",
    )

    dev_plus = m.addVars(base_keys, vtype=GRB.CONTINUOUS, lb=0, name="workload_dev_plus")
    dev_minus = m.addVars(base_keys, vtype=GRB.CONTINUOUS, lb=0, name="workload_dev_minus")
    risk_slack = m.addVars(scenario_names, vtype=GRB.CONTINUOUS, lb=0, name="risk_slack")

    # -------------------------
    # Objective function
    # -------------------------
    pairing_cost = gp.quicksum(float(row.cost) * x[row.pairing_id] for row in pairings.itertuples(index=False))
    reserve_cost = gp.quicksum(420.0 * reserve[r] for r in range(len(reserve_keys)))
    fairness_cost = gp.quicksum(65.0 * (dev_plus[b] + dev_minus[b]) for b in base_keys)
    risk_slack_cost = gp.quicksum(900.0 * risk_slack[s] for s in scenario_names)

    m.setObjective(pairing_cost + reserve_cost + fairness_cost + risk_slack_cost, GRB.MINIMIZE)

    # -------------------------
    # 1) Set partitioning: every flight must be covered exactly once.
    # -------------------------
    for f in flights.flight_id:
        covering_pairings = legs_by_flight.get(f, [])
        if not covering_pairings:
            raise ValueError(
                f"Flight {f} has no candidate pairing. Re-run src/build_pairings.py "
                "or add recovery pairings."
            )
        m.addConstr(
            gp.quicksum(x[p] for p in covering_pairings) == 1,
            name=f"cover_{f}",
        )

    # -------------------------
    # 2) Crew availability by start day/base/qualification pool.
    #    Reserve crews consume the same capacity as selected pairings.
    # -------------------------
    for ridx, (day, base, qual) in enumerate(reserve_keys):
        avail_row = availability[
            (availability.day == day)
            & (availability.base == base)
            & (availability.qualification_pool == qual)
        ].iloc[0]

        available_crews = int(avail_row.available_crews)
        minimum_reserve = int(avail_row.minimum_reserve)

        eligible_pairings = pairings[
            (pairings.start_day == day)
            & (pairings.base == base)
            & (pairings.qualification_pool.isin([qual, "BOTH"]))
        ]["pairing_id"].tolist()

        m.addConstr(
            gp.quicksum(x[p] for p in eligible_pairings) + reserve[ridx] <= available_crews,
            name=f"crew_avail_D{day}_{base}_{qual}",
        )
        m.addConstr(
            reserve[ridx] >= minimum_reserve,
            name=f"min_reserve_D{day}_{base}_{qual}",
        )

    # -------------------------
    # 3) Hotel capacity for overnight pairings.
    # -------------------------
    hotel_pairings = pairings[
        (pairings.hotel_nights > 0)
        & (pairings.overnight_station.notna())
        & (pairings.overnight_station != "")
    ]

    for (station, hotel_day), group in hotel_pairings.groupby(["overnight_station", "hotel_day"]):
        cap = int(airports.loc[airports.airport == station, "hotel_rooms_per_night"].iloc[0])
        m.addConstr(
            gp.quicksum(x[p] for p in group.pairing_id) <= cap,
            name=f"hotel_{station}_D{int(hotel_day)}",
        )

    # -------------------------
    # 4) Deadhead / repositioning budget by base and start day.
    # -------------------------
    for (base, day), group in pairings.groupby(["base", "start_day"]):
        m.addConstr(
            gp.quicksum((1.0 if float(row.deadhead_min) > 0 else 0.0) * x[row.pairing_id]
                        for row in group.itertuples(index=False)) <= 10,
            name=f"deadhead_budget_{base}_D{int(day)}",
        )

    # -------------------------
    # 5) Workload balance by crew base.
    #    These are soft constraints using absolute-deviation variables.
    # -------------------------
    target_duty = 0.25 * float(pairings.duty_min.median()) * len(flights) / 1.8

    for base in base_keys:
        base_pairings = pairings[pairings.base == base]

        # Duty can exceed target by dev_plus.
        m.addConstr(
            gp.quicksum(float(row.duty_min) * x[row.pairing_id] for row in base_pairings.itertuples(index=False))
            - dev_plus[base]
            <= target_duty,
            name=f"workload_upper_{base}",
        )

        # Duty should not fall too far below target; dev_minus absorbs shortfall.
        m.addConstr(
            gp.quicksum(float(row.duty_min) * x[row.pairing_id] for row in base_pairings.itertuples(index=False))
            + dev_minus[base]
            >= 0.65 * target_duty,
            name=f"workload_lower_{base}",
        )

    # -------------------------
    # 6) Stochastic disruption exposure budget by delay scenario.
    #    Risk slack is expensive, so the solver avoids fragile pairings unless necessary.
    # -------------------------
    for s in scenario_names:
        risk_col = f"risk_{s}"
        if risk_col not in pairings.columns:
            raise ValueError(f"Missing scenario risk column: {risk_col}")

        budget = 900.0 if s == "S0_nominal" else 1450.0
        m.addConstr(
            gp.quicksum(float(row._asdict()[risk_col]) * x[row.pairing_id] for row in pairings.itertuples(index=False))
            - risk_slack[s]
            <= budget,
            name=f"risk_budget_{s}",
        )

    # Solve.
    m.optimize()

    if m.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        print("Model is infeasible or unbounded. Computing IIS...", file=sys.stderr)
        m.computeIIS()
        iis_path = OUT / "gurobi_infeasibility.ilp"
        m.write(str(iis_path))
        raise RuntimeError(f"No feasible solution. IIS written to {iis_path}")

    if m.SolCount == 0:
        raise RuntimeError(f"Gurobi did not find a feasible solution. Status code: {m.Status}")

    # -------------------------
    # Extract solution and write outputs.
    # -------------------------
    selected_ids = [p for p in pairing_ids if x[p].X > 0.5]
    selected = pairings[pairings.pairing_id.isin(selected_ids)].copy().sort_values(["start_day", "base", "cost"])
    selected_legs = legs[legs.pairing_id.isin(selected_ids)].merge(flights, on="flight_id", how="left")

    reserve_sol = pd.DataFrame(reserve_keys, columns=["day", "base", "qualification_pool"])
    reserve_sol["reserve_crews"] = [int(round(reserve[r].X)) for r in range(len(reserve_keys))]

    base_workload = selected.groupby("base").agg(
        selected_pairings=("pairing_id", "count"),
        duty_min=("duty_min", "sum"),
        block_min=("block_min", "sum"),
        deadhead_min=("deadhead_min", "sum"),
        avg_expected_delay_risk=("expected_delay_risk", "mean"),
    ).reset_index()
    base_workload["duty_hours"] = (base_workload.duty_min / 60).round(2)

    coverage = selected_legs.groupby("flight_id").pairing_id.nunique().reset_index(name="times_covered")
    coverage_check = flights[["flight_id"]].merge(coverage, on="flight_id", how="left").fillna({"times_covered": 0})
    coverage_check["times_covered"] = coverage_check["times_covered"].astype(int)

    risk_summary_rows = []
    for s in scenario_names:
        risk_col = f"risk_{s}"
        risk = float(pairings.loc[pairings.pairing_id.isin(selected_ids), risk_col].sum())
        risk_summary_rows.append(
            {
                "scenario": s,
                "probability": scenario_probs[s],
                "selected_pairing_risk_min": round(risk, 2),
                "risk_slack_min": round(float(risk_slack[s].X), 2),
            }
        )
    risk_summary = pd.DataFrame(risk_summary_rows)

    summary = pd.DataFrame(
        [
            {"metric": "solver", "value": "Gurobi"},
            {"metric": "solver_status", "value": int(m.Status)},
            {"metric": "solver_status_text", "value": status_text(m.Status)},
            {"metric": "objective_value", "value": round(float(m.ObjVal), 2)},
            {"metric": "best_bound", "value": round(float(m.ObjBound), 2)},
            {"metric": "mip_gap", "value": round(float(m.MIPGap), 6) if m.SolCount > 0 else None},
            {"metric": "runtime_seconds", "value": round(float(m.Runtime), 2)},
            {"metric": "candidate_pairings", "value": len(pairings)},
            {"metric": "selected_pairings", "value": len(selected)},
            {"metric": "flights", "value": len(flights)},
            {"metric": "all_flights_covered_exactly_once", "value": bool((coverage_check.times_covered == 1).all())},
            {"metric": "recovery_single_pairings_selected", "value": int((selected.pairing_type == "recovery_single").sum())},
            {"metric": "round_trip_pairings_selected", "value": int((selected.pairing_type == "round_trip").sum())},
            {"metric": "overnight_pairings_selected", "value": int((selected.pairing_type == "overnight").sum())},
            {"metric": "three_leg_pairings_selected", "value": int((selected.pairing_type == "three_leg").sum())},
            {"metric": "total_reserve_crews", "value": int(reserve_sol.reserve_crews.sum())},
            {"metric": "max_base_duty_hours", "value": round(float(base_workload.duty_hours.max()), 2)},
            {"metric": "min_base_duty_hours", "value": round(float(base_workload.duty_hours.min()), 2)},
            {"metric": "total_deadhead_hours", "value": round(float(selected.deadhead_min.sum() / 60), 2)},
            {"metric": "total_expected_delay_risk_min", "value": round(float(selected.expected_delay_risk.sum()), 2)},
        ]
    )

    selected.to_csv(OUT / "selected_pairings_gurobi.csv", index=False)
    selected_legs.to_csv(OUT / "selected_pairing_legs_gurobi.csv", index=False)
    reserve_sol.to_csv(OUT / "reserve_plan_gurobi.csv", index=False)
    base_workload.to_csv(OUT / "base_workload_summary_gurobi.csv", index=False)
    coverage_check.to_csv(OUT / "flight_coverage_check_gurobi.csv", index=False)
    risk_summary.to_csv(OUT / "scenario_risk_summary_gurobi.csv", index=False)
    summary.to_csv(OUT / "solution_summary_gurobi.csv", index=False)

    # Also create a single Excel workbook for easy interview/demo review.
    excel_path = OUT / "airline_crew_scheduling_gurobi_outputs.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        selected.to_excel(writer, sheet_name="selected_pairings", index=False)
        selected_legs.to_excel(writer, sheet_name="selected_legs", index=False)
        reserve_sol.to_excel(writer, sheet_name="reserve_plan", index=False)
        base_workload.to_excel(writer, sheet_name="base_workload", index=False)
        coverage_check.to_excel(writer, sheet_name="coverage_check", index=False)
        risk_summary.to_excel(writer, sheet_name="scenario_risk", index=False)

    print("\nGurobi solution summary")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {OUT}")
    print(f"Excel workbook: {excel_path}")

    return {
        "summary": summary,
        "selected_pairings": selected,
        "selected_legs": selected_legs,
        "reserve_plan": reserve_sol,
        "base_workload": base_workload,
        "coverage_check": coverage_check,
        "scenario_risk": risk_summary,
    }


def status_text(status_code: int) -> str:
    """Return a readable Gurobi status label."""
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return mapping.get(status_code, f"STATUS_{status_code}")


if __name__ == "__main__":
    solve_with_gurobi(time_limit=120, mip_gap=0.01, verbose=True)
