from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

BASES = ["ATL", "DFW", "ORD", "JFK"]
OTHER_AIRPORTS = [
    "BOS", "MIA", "CLT", "PHL", "DCA", "IAD", "RDU", "BNA", "MSY", "TPA", "MCO", "FLL",
    "DEN", "PHX", "LAS", "LAX", "SFO", "SEA", "SAN", "SLC", "MSP", "DTW", "CLE", "CMH",
    "PIT", "STL", "MCI", "AUS", "SAT", "IAH", "HOU", "SAV"
]
AIRPORTS = BASES + OTHER_AIRPORTS[:32]
assert len(AIRPORTS) == 36

REGION = {
    "ATL":"SE", "DFW":"TX", "ORD":"MW", "JFK":"NE", "BOS":"NE", "MIA":"SE", "CLT":"SE", "PHL":"NE", "DCA":"NE", "IAD":"NE",
    "RDU":"SE", "BNA":"SE", "MSY":"SE", "TPA":"SE", "MCO":"SE", "FLL":"SE", "DEN":"W", "PHX":"W", "LAS":"W", "LAX":"W",
    "SFO":"W", "SEA":"W", "SAN":"W", "SLC":"W", "MSP":"MW", "DTW":"MW", "CLE":"MW", "CMH":"MW", "PIT":"MW", "STL":"MW",
    "MCI":"MW", "AUS":"TX", "SAT":"TX", "IAH":"TX", "HOU":"TX", "SAV":"SE"
}

def minutes_to_clock(m: int) -> str:
    d = m // 1440
    rem = m % 1440
    return f"D{d+1} {rem//60:02d}:{rem%60:02d}"

# Airport attributes
airport_rows = []
for a in AIRPORTS:
    is_base = a in BASES
    airport_rows.append({
        "airport": a,
        "region": REGION[a],
        "is_base": int(is_base),
        "min_turn_min": int(RNG.choice([35, 40, 45, 50], p=[.2,.4,.3,.1])),
        "curfew_start_min": 23*60 if not is_base and RNG.random() < .25 else -1,
        "curfew_end_min": 5*60 if not is_base and RNG.random() < .25 else -1,
        "hotel_rooms_per_night": int(RNG.integers(4, 12) if not is_base else 999),
        "deadhead_seat_cap_per_day": int(RNG.integers(4, 14))
    })
airports = pd.DataFrame(airport_rows)
airports.to_csv(RAW / "airports.csv", index=False)

# Approximate block-time matrix based on regions; intentionally asymmetric noise.
region_time = {
    ("NE","NE"):70,("NE","SE"):120,("NE","MW"):125,("NE","TX"):190,("NE","W"):330,
    ("SE","SE"):75,("SE","MW"):120,("SE","TX"):135,("SE","W"):285,
    ("MW","MW"):70,("MW","TX"):130,("MW","W"):225,
    ("TX","TX"):60,("TX","W"):190,("W","W"):80
}
def block_time(o: str, d: str) -> int:
    if o == d:
        return 0
    ro, rd = REGION[o], REGION[d]
    key = (ro, rd) if (ro, rd) in region_time else (rd, ro)
    return int(max(45, region_time[key] + RNG.normal(0, 18)))

travel_rows = []
for o in AIRPORTS:
    for d in AIRPORTS:
        if o != d:
            travel_rows.append({"origin": o, "destination": d, "block_min": block_time(o, d)})
pd.DataFrame(travel_rows).to_csv(RAW / "block_time_matrix.csv", index=False)

# Create structured flight banks to make feasible multi-leg pairings.
spokes_by_base = {
    "ATL": ["CLT","RDU","BNA","MCO","TPA","MIA","FLL","SAV","MSY"],
    "DFW": ["AUS","SAT","IAH","HOU","MCI","DEN","PHX","LAS"],
    "ORD": ["MSP","DTW","CLE","CMH","PIT","STL","MCI","DEN"],
    "JFK": ["BOS","PHL","DCA","IAD","RDU","CLT","MIA","FLL"]
}
equipment_types = ["A320", "B737", "E175"]
qualification_required = {"A320":"NB", "B737":"NB", "E175":"RJ"}

flights = []
fid = 1
for day in range(3):
    day0 = day * 1440
    for base in BASES:
        spokes = spokes_by_base[base]
        n_cycles = 8 if day < 2 else 9  # 32+32+36 = 100 flights as two-leg cycles count*2? adjust below
        for c in range(n_cycles):
            sp = spokes[c % len(spokes)]
            dep1 = day0 + int(RNG.choice([6*60,7*60,8*60,10*60,12*60,14*60,16*60,18*60]) + RNG.integers(-20, 25))
            bt1 = block_time(base, sp)
            arr1 = dep1 + bt1
            turn = int(RNG.integers(45, 110))
            dep2 = arr1 + turn
            bt2 = block_time(sp, base)
            equip = str(RNG.choice(equipment_types, p=[.42,.38,.20]))
            demand_weight = float(np.round(RNG.uniform(0.6, 1.8), 2))
            flights.append({
                "flight_id": f"F{fid:03d}", "day": day+1, "origin": base, "destination": sp,
                "departure_min": dep1, "arrival_min": arr1, "departure_time": minutes_to_clock(dep1), "arrival_time": minutes_to_clock(arr1),
                "block_min": bt1, "equipment": equip, "qual_required": qualification_required[equip],
                "priority": int(RNG.choice([1,2,3], p=[.15,.55,.30])), "demand_weight": demand_weight
            }); fid += 1
            flights.append({
                "flight_id": f"F{fid:03d}", "day": day+1, "origin": sp, "destination": base,
                "departure_min": dep2, "arrival_min": dep2+bt2, "departure_time": minutes_to_clock(dep2), "arrival_time": minutes_to_clock(dep2+bt2),
                "block_min": bt2, "equipment": equip, "qual_required": qualification_required[equip],
                "priority": int(RNG.choice([1,2,3], p=[.15,.55,.30])), "demand_weight": demand_weight
            }); fid += 1
# Keep exactly 100, then deliberately convert a subset of late turns into overnight turns.
# This forces the optimization to consider hotel capacity, rest rules, and two-day pairings.
flights = pd.DataFrame(flights).sort_values("departure_min").head(100).reset_index(drop=True)
for day in [1, 2]:
    late_outbounds = flights[(flights.day == day) & (flights.origin.isin(BASES)) & (flights.departure_min % 1440 >= 15*60)].head(4)
    for idx, outb in late_outbounds.iterrows():
        mask = (flights.day == day) & (flights.origin == outb.destination) & (flights.destination == outb.origin) & (flights.departure_min > outb.arrival_min)
        if mask.any():
            ret_idx = flights[mask].index[0]
            new_dep = day*1440 + int(RNG.choice([6*60+30, 7*60+15, 8*60])) + int(RNG.integers(-10, 10))
            bt = block_time(outb.destination, outb.origin)
            flights.loc[ret_idx, "day"] = day + 1
            flights.loc[ret_idx, "departure_min"] = new_dep
            flights.loc[ret_idx, "arrival_min"] = new_dep + bt
            flights.loc[ret_idx, "departure_time"] = minutes_to_clock(new_dep)
            flights.loc[ret_idx, "arrival_time"] = minutes_to_clock(new_dep + bt)
            flights.loc[ret_idx, "block_min"] = bt
flights = flights.sort_values("departure_min").reset_index(drop=True)
flights["flight_id"] = [f"F{i+1:03d}" for i in range(len(flights))]
flights.to_csv(RAW / "flights.csv", index=False)

# Crew members and base/qualification availability.
crew_rows = []
crew_id = 1
base_counts = {"ATL":34,"DFW":31,"ORD":30,"JFK":30}  # total 125
for base, count in base_counts.items():
    for _ in range(count):
        qual = str(RNG.choice(["NB", "RJ", "BOTH"], p=[.58,.18,.24]))
        seniority = int(RNG.integers(1, 26))
        max_duty = int(RNG.choice([600, 660, 720], p=[.2,.55,.25]))
        crew_rows.append({
            "crew_id": f"C{crew_id:03d}", "base": base, "qualification": qual,
            "seniority_years": seniority, "max_duty_min": max_duty,
            "prefers_day_trips": int(RNG.random() < .62),
            "max_pairings_horizon": int(RNG.choice([1,2], p=[.78,.22]))
        })
        crew_id += 1
crew = pd.DataFrame(crew_rows)
crew.to_csv(RAW / "crew.csv", index=False)

# Aggregate crew availability by base, qualification, and day.
avail_rows = []
for day in [1,2,3]:
    for base in BASES:
        subset = crew[crew.base == base]
        nb = int(((subset.qualification == "NB") | (subset.qualification == "BOTH")).sum() * RNG.uniform(.62, .76))
        rj = int(((subset.qualification == "RJ") | (subset.qualification == "BOTH")).sum() * RNG.uniform(.58, .72))
        both = int((subset.qualification == "BOTH").sum() * RNG.uniform(.45, .65))
        reserve_req = int(RNG.integers(2, 5))
        avail_rows += [
            {"day":day,"base":base,"qualification_pool":"NB","available_crews":nb,"minimum_reserve":reserve_req},
            {"day":day,"base":base,"qualification_pool":"RJ","available_crews":rj,"minimum_reserve":max(1, reserve_req-1)},
        ]
availability = pd.DataFrame(avail_rows)
availability.to_csv(RAW / "crew_availability.csv", index=False)

# Crew preference penalty matrix by base-region, day, and duty type.
pref_rows = []
for base in BASES:
    for dest_region in sorted(set(REGION.values())):
        for day in [1,2,3]:
            pref_rows.append({
                "base": base,
                "overnight_region": dest_region,
                "day": day,
                "preference_penalty": int(RNG.integers(0, 80) + (25 if REGION[base] != dest_region else 0))
            })
pd.DataFrame(pref_rows).to_csv(RAW / "preference_penalties.csv", index=False)

# Disruption scenarios: probability and flight-level delay shock. Used for expected overtime/misconnect risk.
scenario_rows = []
scenario_probs = {"S0_nominal":0.55,"S1_weather_NE":0.16,"S2_weather_SE":0.13,"S3_ATC_MW":0.09,"S4_west_coast":0.07}
for s, prob in scenario_probs.items():
    for _, f in flights.iterrows():
        region = REGION[f.destination]
        base_delay = 8
        if s == "S1_weather_NE" and region == "NE": base_delay = 65
        elif s == "S2_weather_SE" and region == "SE": base_delay = 55
        elif s == "S3_ATC_MW" and region == "MW": base_delay = 50
        elif s == "S4_west_coast" and region == "W": base_delay = 70
        delay = int(max(0, RNG.normal(base_delay, 12)))
        scenario_rows.append({"scenario":s,"probability":prob,"flight_id":f.flight_id,"arrival_delay_min":delay})
pd.DataFrame(scenario_rows).to_csv(RAW / "delay_scenarios.csv", index=False)

rules = pd.DataFrame([
    {"parameter":"min_connection_min", "value":45, "description":"Minimum connection/turn time between operated legs"},
    {"parameter":"max_sit_min", "value":240, "description":"Maximum sit time inside a duty period"},
    {"parameter":"max_duty_min", "value":720, "description":"Maximum duty elapsed time for a pairing duty"},
    {"parameter":"max_block_min", "value":540, "description":"Maximum flying block time per duty"},
    {"parameter":"brief_min", "value":45, "description":"Pre-flight report/brief time"},
    {"parameter":"debrief_min", "value":30, "description":"Post-duty debrief time"},
    {"parameter":"min_rest_min", "value":600, "description":"Minimum overnight rest"},
    {"parameter":"max_legs_per_pairing", "value":4, "description":"Pairing generation limit"},
    {"parameter":"max_pairing_span_days", "value":2, "description":"Maximum generated pairing duration"},
])
rules.to_csv(RAW / "duty_rules.csv", index=False)

print(f"Generated {len(flights)} flights, {len(AIRPORTS)} airports, {len(crew)} crew members.")
