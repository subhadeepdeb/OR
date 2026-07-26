
"""

Models included:
1) Base MCLP: maximize covered demand under budget, with assignment, station capacity, and grid connection feasibility.
2) Equity MCLP: maximize equity-weighted covered demand and enforce minimum high-TDI coverage.
3) Base FCLP: minimize station fixed cost + user travel cost + grid connection/transmission cost while serving required demand.
4) Equity FCLP: FCLP with high-TDI service requirement and optional equity-weighted lost-demand penalty.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class EVConfig:
    # Coverage/service assumptions
    service_radius_miles: float = 3.0
    max_grid_connection_miles: float = 10.0

    # Budget and costs
    budget: float = 800_000.0
    fixed_station_cost: float = 20_000.0
    charger_port_cost: float = 18_000.0
    grid_connection_cost_per_mile: float = 10_000.0
    user_travel_cost_per_mile: float = 1.0

    # Charging station capacity assumptions
    # One "port" can serve this many demand units over the planning period.
    demand_units_per_port: float = 35.0
    min_ports_per_open_station: int = 2
    max_ports_per_open_station: int = 12

    # Power conversion
    # Level-3 DC fast charger; adjust as needed.
    charger_power_mw: float = 0.150

    # Only a fraction of listed substation final capacity is assumed available for EV expansion.
    # This avoids unrealistically using full grid capacity.
    grid_available_fraction: float = 0.02

    # Service requirements
    required_demand_fraction: float = 0.75
    high_tdi_quantile: float = 0.75
    min_high_tdi_coverage_fraction: float = 0.70

    # Equity weighting
    equity_weight_strength: float = 1.0 

    # Solver
    mip_gap: float = 0.01
    time_limit_seconds: int = 300


def _read_first_sheet(path: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def load_ev_data(data_dir: str = ".") -> Dict[str, Any]:
    
    files = {
        "candidates": os.path.join(data_dir, "TrialDataNCDOT.xlsx"),
        "demand": os.path.join(data_dir, "Demand Calculation.xlsx"),
        "equity": os.path.join(data_dir, "Data.xlsx"),
        "distances": os.path.join(data_dir, "distance_matrice.xlsx"),
        "substations": os.path.join(data_dir, "power_substation.xlsx"),
        "substation_capacity": os.path.join(data_dir, "Max capacity of power_substations.xlsx"),
        "raleigh_locations": os.path.join(data_dir, "raleigh_locations.csv"),
        "all_nodes": os.path.join(data_dir, "nodes_locations.csv"),
    }

    missing = [name for name, path in files.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing expected files: {missing}")

    candidates = _read_first_sheet(files["candidates"]).copy()
    demand_df = _read_first_sheet(files["demand"]).copy()
    equity_df = pd.read_excel(files["equity"], sheet_name="Data").copy()

    travel = pd.read_excel(files["distances"], sheet_name="Travel_Costs", index_col=0)
    transmission = pd.read_excel(files["distances"], sheet_name="Transmission_Costs", index_col=0)

    substations = _read_first_sheet(files["substations"]).copy()
    subcap = pd.read_excel(files["substation_capacity"], sheet_name="in").copy()
    raleigh_locations = pd.read_csv(files["raleigh_locations"])
    all_nodes = pd.read_csv(files["all_nodes"])

    # Basic validation
    n_i = len(candidates)
    n_j = len(demand_df)
    n_k = len(substations)

    if travel.shape != (n_i, n_j):
        raise ValueError(f"Travel matrix shape {travel.shape} does not match candidates x demand {(n_i, n_j)}")
    if transmission.shape != (n_k, n_i):
        raise ValueError(f"Transmission matrix shape {transmission.shape} does not match substations x candidates {(n_k, n_i)}")
    if "Calculated Demand" not in demand_df.columns:
        raise ValueError("Demand Calculation.xlsx must contain 'Calculated Demand'")
    if "TDI Score" not in demand_df.columns:
        raise ValueError("Demand Calculation.xlsx must contain 'TDI Score'")
    if "Final maxcap (MW)" not in subcap.columns:
        raise ValueError("Max capacity file must contain 'Final maxcap (MW)'")

    # Clean numeric arrays
    demand = pd.to_numeric(demand_df["Calculated Demand"], errors="coerce").fillna(0).to_numpy(dtype=float)
    tdi = pd.to_numeric(demand_df["TDI Score"], errors="coerce")
    if tdi.isna().any():
        # Fill missing TDI by GEOID match from Data.xlsx if possible, then median
        if "GEOID" in demand_df.columns and "GEOID" in equity_df.columns and "TDI" in equity_df.columns:
            tdi_map = equity_df.set_index("GEOID")["TDI"]
            tdi = demand_df["GEOID"].map(tdi_map).fillna(tdi)
        tdi = tdi.fillna(tdi.median())
    tdi = tdi.to_numpy(dtype=float)

    travel_miles = travel.to_numpy(dtype=float)
    transmission_miles = transmission.to_numpy(dtype=float)
    substation_capacity_mw = pd.to_numeric(subcap["Final maxcap (MW)"], errors="coerce").fillna(0).to_numpy(dtype=float)

    # Standardized sets
    I = list(range(n_i))  # candidate station sites
    J = list(range(n_j))  # demand zones
    K = list(range(n_k))  # substations

    return {
        "files": files,
        "candidates": candidates,
        "demand_df": demand_df,
        "equity_df": equity_df,
        "travel": travel,
        "transmission": transmission,
        "substations": substations,
        "subcap": subcap,
        "raleigh_locations": raleigh_locations,
        "all_nodes": all_nodes,
        "I": I,
        "J": J,
        "K": K,
        "demand": demand,
        "tdi": tdi,
        "travel_miles": travel_miles,
        "transmission_miles": transmission_miles,
        "substation_capacity_mw": substation_capacity_mw,
    }


def summarize_data(data: Dict[str, Any]) -> pd.DataFrame:
    demand = data["demand"]
    tdi = data["tdi"]
    travel = data["travel_miles"]
    trans = data["transmission_miles"]
    cap = data["substation_capacity_mw"]
    rows = [
        ("candidate_sites", len(data["I"])),
        ("demand_zones", len(data["J"])),
        ("substations", len(data["K"])),
        ("total_demand", demand.sum()),
        ("min_demand", demand.min()),
        ("max_demand", demand.max()),
        ("min_tdi", tdi.min()),
        ("max_tdi", tdi.max()),
        ("max_travel_miles", travel.max()),
        ("max_transmission_miles", trans.max()),
        ("total_substation_final_capacity_mw", cap.sum()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def require_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
        return gp, GRB
    except Exception as exc:
        raise ImportError(
            "This completed project uses gurobipy. Install Gurobi/gurobipy and activate a license, "
            "or translate the linear models to Pyomo/PuLP."
        ) from exc


def _derived_parameters(data: Dict[str, Any], cfg: EVConfig) -> Dict[str, Any]:
    I, J, K = data["I"], data["J"], data["K"]
    demand = data["demand"]
    tdi = data["tdi"]
    travel = data["travel_miles"]
    trans = data["transmission_miles"]

    # Station capacity in demand units if opened at max ports.
    station_capacity_demand = cfg.max_ports_per_open_station * cfg.demand_units_per_port
    min_station_capacity_demand = cfg.min_ports_per_open_station * cfg.demand_units_per_port

    # Power required per demand unit at station.
    # If one port is 0.150 MW and one port serves demand_units_per_port, then MW per demand unit:
    mw_per_demand_unit = cfg.charger_power_mw / cfg.demand_units_per_port

    available_substation_capacity_mw = data["substation_capacity_mw"] * cfg.grid_available_fraction

    cover = {(i, j): 1 if travel[i, j] <= cfg.service_radius_miles else 0 for i in I for j in J}
    grid_feasible = {(k, i): 1 if trans[k, i] <= cfg.max_grid_connection_miles else 0 for k in K for i in I}

    tdi_min, tdi_max = float(np.min(tdi)), float(np.max(tdi))
    tdi_norm = (tdi - tdi_min) / (tdi_max - tdi_min + 1e-9)
    equity_weight = 1.0 + cfg.equity_weight_strength * tdi_norm
    high_tdi_cut = float(np.quantile(tdi, cfg.high_tdi_quantile))
    high_tdi_zones = [j for j in J if tdi[j] >= high_tdi_cut]

    return {
        "station_capacity_demand": station_capacity_demand,
        "min_station_capacity_demand": min_station_capacity_demand,
        "mw_per_demand_unit": mw_per_demand_unit,
        "available_substation_capacity_mw": available_substation_capacity_mw,
        "cover": cover,
        "grid_feasible": grid_feasible,
        "equity_weight": equity_weight,
        "high_tdi_cut": high_tdi_cut,
        "high_tdi_zones": high_tdi_zones,
    }


def solve_base_mclp(data: Dict[str, Any], cfg: EVConfig = EVConfig()):
    
    gp, GRB = require_gurobi()
    I, J, K = data["I"], data["J"], data["K"]
    d, travel, trans = data["demand"], data["travel_miles"], data["transmission_miles"]
    par = _derived_parameters(data, cfg)

    m = gp.Model("Base_MCLP_EV_Infrastructure")
    m.Params.MIPGap = cfg.mip_gap
    m.Params.TimeLimit = cfg.time_limit_seconds

    x = m.addVars(I, vtype=GRB.BINARY, name="open_station")
    assign = m.addVars(I, J, vtype=GRB.BINARY, name="assign_demand")
    y = m.addVars(J, vtype=GRB.BINARY, name="covered_demand")
    g = m.addVars(K, I, vtype=GRB.BINARY, name="grid_connection")
    p = m.addVars(K, I, lb=0.0, vtype=GRB.CONTINUOUS, name="power_mw")
    ports = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=cfg.max_ports_per_open_station, name="charger_ports")

    m.setObjective(gp.quicksum(d[j] * y[j] for j in J), GRB.MAXIMIZE)

    # Budget: station fixed + ports + one-time grid line cost.
    m.addConstr(
        gp.quicksum(cfg.fixed_station_cost * x[i] + cfg.charger_port_cost * ports[i] for i in I)
        + gp.quicksum(cfg.grid_connection_cost_per_mile * trans[k, i] * g[k, i] for k in K for i in I)
        <= cfg.budget,
        name="budget"
    )

    for i in I:
        m.addConstr(ports[i] <= cfg.max_ports_per_open_station * x[i], name=f"ports_upper_{i}")
        m.addConstr(ports[i] >= cfg.min_ports_per_open_station * x[i], name=f"ports_lower_{i}")
        # Each opened station must connect to exactly one feasible substation.
        m.addConstr(gp.quicksum(g[k, i] for k in K) == x[i], name=f"one_grid_connection_if_open_{i}")
        # Demand assigned to station cannot exceed installed-port capacity.
        m.addConstr(gp.quicksum(d[j] * assign[i, j] for j in J) <= cfg.demand_units_per_port * ports[i],
                    name=f"station_demand_capacity_{i}")
        # Supplied MW must support installed chargers.
        m.addConstr(gp.quicksum(p[k, i] for k in K) >= cfg.charger_power_mw * ports[i],
                    name=f"station_power_requirement_{i}")

    for j in J:
        # A demand zone is covered if assigned to exactly one open covering station.
        m.addConstr(gp.quicksum(assign[i, j] for i in I) == y[j], name=f"coverage_assignment_{j}")

    for i in I:
        for j in J:
            m.addConstr(assign[i, j] <= x[i], name=f"assign_only_if_open_{i}_{j}")
            m.addConstr(assign[i, j] <= par["cover"][i, j], name=f"assign_only_if_within_radius_{i}_{j}")

    for k in K:
        m.addConstr(gp.quicksum(p[k, i] for i in I) <= par["available_substation_capacity_mw"][k],
                    name=f"substation_capacity_{k}")
        for i in I:
            m.addConstr(g[k, i] <= par["grid_feasible"][k, i], name=f"grid_distance_feasibility_{k}_{i}")
            m.addConstr(g[k, i] <= x[i], name=f"grid_only_if_open_{k}_{i}")
            m.addConstr(p[k, i] <= par["available_substation_capacity_mw"][k] * g[k, i],
                        name=f"power_only_if_connected_{k}_{i}")

    m.optimize()
    return m, {"x": x, "assign": assign, "covered": y, "grid_connection": g, "power_mw": p, "ports": ports}


def solve_equity_mclp(data: Dict[str, Any], cfg: EVConfig = EVConfig()):
    
    gp, GRB = require_gurobi()
    I, J, K = data["I"], data["J"], data["K"]
    d, travel, trans = data["demand"], data["travel_miles"], data["transmission_miles"]
    par = _derived_parameters(data, cfg)
    w = par["equity_weight"]
    H = par["high_tdi_zones"]

    m = gp.Model("Equity_MCLP_EV_Infrastructure")
    m.Params.MIPGap = cfg.mip_gap
    m.Params.TimeLimit = cfg.time_limit_seconds

    x = m.addVars(I, vtype=GRB.BINARY, name="open_station")
    assign = m.addVars(I, J, vtype=GRB.BINARY, name="assign_demand")
    y = m.addVars(J, vtype=GRB.BINARY, name="covered_demand")
    g = m.addVars(K, I, vtype=GRB.BINARY, name="grid_connection")
    p = m.addVars(K, I, lb=0.0, vtype=GRB.CONTINUOUS, name="power_mw")
    ports = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=cfg.max_ports_per_open_station, name="charger_ports")

    m.setObjective(gp.quicksum(w[j] * d[j] * y[j] for j in J), GRB.MAXIMIZE)

    m.addConstr(
        gp.quicksum(cfg.fixed_station_cost * x[i] + cfg.charger_port_cost * ports[i] for i in I)
        + gp.quicksum(cfg.grid_connection_cost_per_mile * trans[k, i] * g[k, i] for k in K for i in I)
        <= cfg.budget,
        name="budget"
    )

    if H:
        m.addConstr(
            gp.quicksum(d[j] * y[j] for j in H)
            >= cfg.min_high_tdi_coverage_fraction * sum(d[j] for j in H),
            name="minimum_high_tdi_coverage"
        )

    for i in I:
        m.addConstr(ports[i] <= cfg.max_ports_per_open_station * x[i], name=f"ports_upper_{i}")
        m.addConstr(ports[i] >= cfg.min_ports_per_open_station * x[i], name=f"ports_lower_{i}")
        m.addConstr(gp.quicksum(g[k, i] for k in K) == x[i], name=f"one_grid_connection_if_open_{i}")
        m.addConstr(gp.quicksum(d[j] * assign[i, j] for j in J) <= cfg.demand_units_per_port * ports[i],
                    name=f"station_demand_capacity_{i}")
        m.addConstr(gp.quicksum(p[k, i] for k in K) >= cfg.charger_power_mw * ports[i],
                    name=f"station_power_requirement_{i}")

    for j in J:
        m.addConstr(gp.quicksum(assign[i, j] for i in I) == y[j], name=f"coverage_assignment_{j}")

    for i in I:
        for j in J:
            m.addConstr(assign[i, j] <= x[i], name=f"assign_only_if_open_{i}_{j}")
            m.addConstr(assign[i, j] <= par["cover"][i, j], name=f"assign_only_if_within_radius_{i}_{j}")

    for k in K:
        m.addConstr(gp.quicksum(p[k, i] for i in I) <= par["available_substation_capacity_mw"][k],
                    name=f"substation_capacity_{k}")
        for i in I:
            m.addConstr(g[k, i] <= par["grid_feasible"][k, i], name=f"grid_distance_feasibility_{k}_{i}")
            m.addConstr(g[k, i] <= x[i], name=f"grid_only_if_open_{k}_{i}")
            m.addConstr(p[k, i] <= par["available_substation_capacity_mw"][k] * g[k, i],
                        name=f"power_only_if_connected_{k}_{i}")

    m.optimize()
    return m, {"x": x, "assign": assign, "covered": y, "grid_connection": g, "power_mw": p, "ports": ports}


def solve_base_fclp(data: Dict[str, Any], cfg: EVConfig = EVConfig()):
    
    gp, GRB = require_gurobi()
    I, J, K = data["I"], data["J"], data["K"]
    d, travel, trans = data["demand"], data["travel_miles"], data["transmission_miles"]
    par = _derived_parameters(data, cfg)

    m = gp.Model("Base_FCLP_EV_Infrastructure")
    m.Params.MIPGap = cfg.mip_gap
    m.Params.TimeLimit = cfg.time_limit_seconds

    x = m.addVars(I, vtype=GRB.BINARY, name="open_station")
    assign = m.addVars(I, J, vtype=GRB.BINARY, name="assign_demand")
    served = m.addVars(J, vtype=GRB.BINARY, name="served_demand")
    g = m.addVars(K, I, vtype=GRB.BINARY, name="grid_connection")
    p = m.addVars(K, I, lb=0.0, vtype=GRB.CONTINUOUS, name="power_mw")
    ports = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=cfg.max_ports_per_open_station, name="charger_ports")

    m.setObjective(
        gp.quicksum(cfg.fixed_station_cost * x[i] + cfg.charger_port_cost * ports[i] for i in I)
        + gp.quicksum(cfg.user_travel_cost_per_mile * d[j] * travel[i, j] * assign[i, j] for i in I for j in J)
        + gp.quicksum(cfg.grid_connection_cost_per_mile * trans[k, i] * g[k, i] for k in K for i in I),
        GRB.MINIMIZE
    )

    m.addConstr(gp.quicksum(d[j] * served[j] for j in J) >= cfg.required_demand_fraction * sum(d),
                name="minimum_total_demand_served")

    for i in I:
        m.addConstr(ports[i] <= cfg.max_ports_per_open_station * x[i], name=f"ports_upper_{i}")
        m.addConstr(ports[i] >= cfg.min_ports_per_open_station * x[i], name=f"ports_lower_{i}")
        m.addConstr(gp.quicksum(g[k, i] for k in K) == x[i], name=f"one_grid_connection_if_open_{i}")
        m.addConstr(gp.quicksum(d[j] * assign[i, j] for j in J) <= cfg.demand_units_per_port * ports[i],
                    name=f"station_demand_capacity_{i}")
        m.addConstr(gp.quicksum(p[k, i] for k in K) >= cfg.charger_power_mw * ports[i],
                    name=f"station_power_requirement_{i}")

    for j in J:
        m.addConstr(gp.quicksum(assign[i, j] for i in I) == served[j], name=f"serve_if_assigned_{j}")

    for i in I:
        for j in J:
            m.addConstr(assign[i, j] <= x[i], name=f"assign_only_if_open_{i}_{j}")
            m.addConstr(assign[i, j] <= par["cover"][i, j], name=f"assign_only_if_within_radius_{i}_{j}")

    for k in K:
        m.addConstr(gp.quicksum(p[k, i] for i in I) <= par["available_substation_capacity_mw"][k],
                    name=f"substation_capacity_{k}")
        for i in I:
            m.addConstr(g[k, i] <= par["grid_feasible"][k, i], name=f"grid_distance_feasibility_{k}_{i}")
            m.addConstr(g[k, i] <= x[i], name=f"grid_only_if_open_{k}_{i}")
            m.addConstr(p[k, i] <= par["available_substation_capacity_mw"][k] * g[k, i],
                        name=f"power_only_if_connected_{k}_{i}")

    m.optimize()
    return m, {"x": x, "assign": assign, "served": served, "grid_connection": g, "power_mw": p, "ports": ports}


def solve_equity_fclp(data: Dict[str, Any], cfg: EVConfig = EVConfig()):
    
    gp, GRB = require_gurobi()
    I, J, K = data["I"], data["J"], data["K"]
    d, travel, trans = data["demand"], data["travel_miles"], data["transmission_miles"]
    par = _derived_parameters(data, cfg)
    H = par["high_tdi_zones"]

    m = gp.Model("Equity_FCLP_EV_Infrastructure")
    m.Params.MIPGap = cfg.mip_gap
    m.Params.TimeLimit = cfg.time_limit_seconds

    x = m.addVars(I, vtype=GRB.BINARY, name="open_station")
    assign = m.addVars(I, J, vtype=GRB.BINARY, name="assign_demand")
    served = m.addVars(J, vtype=GRB.BINARY, name="served_demand")
    g = m.addVars(K, I, vtype=GRB.BINARY, name="grid_connection")
    p = m.addVars(K, I, lb=0.0, vtype=GRB.CONTINUOUS, name="power_mw")
    ports = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub=cfg.max_ports_per_open_station, name="charger_ports")

    m.setObjective(
        gp.quicksum(cfg.fixed_station_cost * x[i] + cfg.charger_port_cost * ports[i] for i in I)
        + gp.quicksum(cfg.user_travel_cost_per_mile * d[j] * travel[i, j] * assign[i, j] for i in I for j in J)
        + gp.quicksum(cfg.grid_connection_cost_per_mile * trans[k, i] * g[k, i] for k in K for i in I),
        GRB.MINIMIZE
    )

    m.addConstr(gp.quicksum(d[j] * served[j] for j in J) >= cfg.required_demand_fraction * sum(d),
                name="minimum_total_demand_served")
    if H:
        m.addConstr(gp.quicksum(d[j] * served[j] for j in H) >= cfg.min_high_tdi_coverage_fraction * sum(d[j] for j in H),
                    name="minimum_high_tdi_demand_served")

    for i in I:
        m.addConstr(ports[i] <= cfg.max_ports_per_open_station * x[i], name=f"ports_upper_{i}")
        m.addConstr(ports[i] >= cfg.min_ports_per_open_station * x[i], name=f"ports_lower_{i}")
        m.addConstr(gp.quicksum(g[k, i] for k in K) == x[i], name=f"one_grid_connection_if_open_{i}")
        m.addConstr(gp.quicksum(d[j] * assign[i, j] for j in J) <= cfg.demand_units_per_port * ports[i],
                    name=f"station_demand_capacity_{i}")
        m.addConstr(gp.quicksum(p[k, i] for k in K) >= cfg.charger_power_mw * ports[i],
                    name=f"station_power_requirement_{i}")

    for j in J:
        m.addConstr(gp.quicksum(assign[i, j] for i in I) == served[j], name=f"serve_if_assigned_{j}")

    for i in I:
        for j in J:
            m.addConstr(assign[i, j] <= x[i], name=f"assign_only_if_open_{i}_{j}")
            m.addConstr(assign[i, j] <= par["cover"][i, j], name=f"assign_only_if_within_radius_{i}_{j}")

    for k in K:
        m.addConstr(gp.quicksum(p[k, i] for i in I) <= par["available_substation_capacity_mw"][k],
                    name=f"substation_capacity_{k}")
        for i in I:
            m.addConstr(g[k, i] <= par["grid_feasible"][k, i], name=f"grid_distance_feasibility_{k}_{i}")
            m.addConstr(g[k, i] <= x[i], name=f"grid_only_if_open_{k}_{i}")
            m.addConstr(p[k, i] <= par["available_substation_capacity_mw"][k] * g[k, i],
                        name=f"power_only_if_connected_{k}_{i}")

    m.optimize()
    return m, {"x": x, "assign": assign, "served": served, "grid_connection": g, "power_mw": p, "ports": ports}


def extract_solution(data: Dict[str, Any], model, vars_dict: Dict[str, Any], cfg: EVConfig = EVConfig()) -> Dict[str, pd.DataFrame]:
    """Extracts station, assignment, and grid solution tables from a solved Gurobi model."""
    if model.SolCount == 0:
        raise ValueError("No feasible solution found.")

    I, J, K = data["I"], data["J"], data["K"]
    demand_df = data["demand_df"]
    cand = data["candidates"]
    sub = data["substations"]
    d = data["demand"]
    tdi = data["tdi"]
    travel = data["travel_miles"]
    trans = data["transmission_miles"]

    x = vars_dict["x"]
    assign = vars_dict["assign"]
    g = vars_dict["grid_connection"]
    ports = vars_dict["ports"]
    p = vars_dict["power_mw"]

    open_rows = []
    for i in I:
        if x[i].X > 0.5:
            assigned_js = [j for j in J if assign[i, j].X > 0.5]
            connected_ks = [k for k in K if g[k, i].X > 0.5]
            k = connected_ks[0] if connected_ks else None
            open_rows.append({
                "station_id": i,
                "latitude": cand.loc[i, "latitude"] if "latitude" in cand.columns else np.nan,
                "longitude": cand.loc[i, "longitude"] if "longitude" in cand.columns else np.nan,
                "ports": round(ports[i].X),
                "assigned_demand_units": sum(d[j] for j in assigned_js),
                "assigned_zones": len(assigned_js),
                "connected_substation": k,
                "substation_latitude": sub.loc[k, "latitude"] if k is not None and "latitude" in sub.columns else np.nan,
                "substation_longitude": sub.loc[k, "longitude"] if k is not None and "longitude" in sub.columns else np.nan,
                "grid_distance_miles": trans[k, i] if k is not None else np.nan,
                "power_mw": sum(p[k2, i].X for k2 in K),
            })

    assign_rows = []
    for i in I:
        for j in J:
            if assign[i, j].X > 0.5:
                assign_rows.append({
                    "station_id": i,
                    "demand_zone": j,
                    "demand": d[j],
                    "tdi_score": tdi[j],
                    "travel_miles": travel[i, j],
                    "demand_latitude": demand_df.loc[j, "latitude"] if "latitude" in demand_df.columns else np.nan,
                    "demand_longitude": demand_df.loc[j, "longitude"] if "longitude" in demand_df.columns else np.nan,
                })

    grid_rows = []
    for k in K:
        supplied = sum(p[k, i].X for i in I)
        if supplied > 1e-6:
            grid_rows.append({
                "substation_id": k,
                "supplied_mw": supplied,
                "latitude": sub.loc[k, "latitude"] if "latitude" in sub.columns else np.nan,
                "longitude": sub.loc[k, "longitude"] if "longitude" in sub.columns else np.nan,
            })

    return {
        "open_stations": pd.DataFrame(open_rows),
        "assignments": pd.DataFrame(assign_rows),
        "grid_supply": pd.DataFrame(grid_rows),
    }


if __name__ == "__main__":
    # Example usage. Put this script in the same folder as the uploaded data files.
    data = load_ev_data(".")
    print(summarize_data(data))

    cfg = EVConfig(
        service_radius_miles=3.0,
        budget=800_000,
        required_demand_fraction=0.75,
        min_high_tdi_coverage_fraction=0.70,
        mip_gap=0.01,
        time_limit_seconds=300,
    )
