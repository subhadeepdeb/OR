from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(123)
BASES = ["ATL", "DFW", "ORD", "JFK"]

flights = pd.read_csv(RAW / "flights.csv")
airports = pd.read_csv(RAW / "airports.csv")
block = pd.read_csv(RAW / "block_time_matrix.csv")
prefs = pd.read_csv(RAW / "preference_penalties.csv")
delays = pd.read_csv(RAW / "delay_scenarios.csv")
rules = pd.read_csv(RAW / "duty_rules.csv").set_index("parameter")["value"].to_dict()

block_dict = {(r.origin, r.destination): int(r.block_min) for r in block.itertuples(index=False)}
hotel_cap = airports.set_index("airport")["hotel_rooms_per_night"].to_dict()
region = airports.set_index("airport")["region"].to_dict()
pref_dict = {(r.base, r.overnight_region, int(r.day)): int(r.preference_penalty) for r in prefs.itertuples(index=False)}
flight_delay = delays.pivot(index="flight_id", columns="scenario", values="arrival_delay_min").fillna(0).to_dict(orient="index")
scenario_probs = delays.groupby("scenario")["probability"].first().to_dict()

MIN_CONN = int(rules["min_connection_min"])
MAX_SIT = int(rules["max_sit_min"])
MAX_DUTY = int(rules["max_duty_min"])
MAX_BLOCK = int(rules["max_block_min"])
BRIEF = int(rules["brief_min"])
DEBRIEF = int(rules["debrief_min"])
MIN_REST = int(rules["min_rest_min"])

def compatible_qual(qual: str, flights_df: pd.DataFrame) -> bool:
    reqs = set(flights_df["qual_required"])
    if qual == "BOTH":
        return True
    return reqs <= {qual}

def legal_sequence(seq: pd.DataFrame) -> tuple[bool, str]:
    seq = seq.sort_values("departure_min")
    if len(seq) == 0:
        return False, "empty"
    # chronological connectivity: crew can operate if destination of previous is origin of next
    for a, b in zip(seq.iloc[:-1].itertuples(), seq.iloc[1:].itertuples()):
        if a.destination != b.origin:
            return False, "not_connected"
        sit = int(b.departure_min - a.arrival_min)
        if sit < MIN_CONN:
            return False, "short_connection"
        if sit > MAX_SIT and int(b.day) == int(a.day):
            return False, "long_sit"
        if int(b.day) > int(a.day):
            rest = sit
            if rest < MIN_REST:
                return False, "insufficient_rest"
    duty = int(seq.departure_min.min() - BRIEF)
    duty_end = int(seq.arrival_min.max() + DEBRIEF)
    elapsed = duty_end - duty
    block_sum = int(seq.block_min.sum())
    if elapsed > MAX_DUTY and seq.day.nunique() == 1:
        return False, "duty_limit"
    if block_sum > MAX_BLOCK:
        return False, "block_limit"
    if len(seq) > 4:
        return False, "too_many_legs"
    if seq.day.max() - seq.day.min() > 1:
        return False, "span_limit"
    return True, "legal"

def pairing_cost(base: str, qual: str, seq: pd.DataFrame, pairing_type: str) -> dict:
    start_airport = seq.iloc[0].origin
    end_airport = seq.iloc[-1].destination
    start_day = int(seq.day.min())
    end_day = int(seq.day.max())
    dh_out = 0 if base == start_airport else block_dict.get((base, start_airport), 999)
    dh_back = 0 if base == end_airport else block_dict.get((end_airport, base), 999)
    # For multi-day pairings, crew duty time is the sum of daily duty periods,
    # not the full elapsed clock span including legal rest/hotel overnight.
    block_min = int(seq.block_min.sum())
    duty_min = 0
    for d, day_seq in seq.groupby("day"):
        day_seq = day_seq.sort_values("departure_min")
        extra_start = dh_out if int(d) == int(seq.day.min()) else 0
        extra_end = dh_back if int(d) == int(seq.day.max()) else 0
        duty_start = int(day_seq.departure_min.min() - BRIEF - extra_start)
        duty_end = int(day_seq.arrival_min.max() + DEBRIEF + extra_end)
        duty_min += duty_end - duty_start
    num_legs = len(seq)
    overnight_station = ""
    hotel_nights = 0
    hotel_key = ""
    hotel_day = 0
    if end_day > start_day or end_airport != base:
        overnight_station = end_airport if end_airport not in BASES else ""
        hotel_nights = 1 if overnight_station else 0
        hotel_day = start_day
        hotel_key = f"{overnight_station}_D{hotel_day}" if overnight_station else ""
    pref_penalty = 0
    if overnight_station:
        pref_penalty = pref_dict.get((base, region[overnight_station], start_day), 35)
    # delay exposure: expected minutes that eat connection buffer or create overtime.
    expected_delay_risk = 0.0
    scenario_exposure = {}
    for s, prob in scenario_probs.items():
        risk = 0.0
        for a, b in zip(seq.iloc[:-1].itertuples(), seq.iloc[1:].itertuples()):
            buffer = max(0, int(b.departure_min - a.arrival_min - MIN_CONN))
            d = flight_delay[a.flight_id][s]
            risk += max(0, d - buffer)
        overtime = max(0, duty_min - MAX_DUTY)
        risk += 0.35 * overtime
        scenario_exposure[s] = float(risk)
        expected_delay_risk += prob * risk
    # realistic cost components
    crew_pay = 2.15 * duty_min + 1.25 * block_min
    deadhead_cost = 3.6 * (dh_out + dh_back)
    hotel_cost = 185 * hotel_nights
    complexity_penalty = 120 * max(0, num_legs - 2)
    recovery_penalty = 1600 if pairing_type == "recovery_single" else 0
    cost = crew_pay + deadhead_cost + hotel_cost + pref_penalty + 18 * expected_delay_risk + complexity_penalty + recovery_penalty
    return {
        "base":base,"qualification_pool":qual,"start_day":start_day,"end_day":end_day,
        "num_legs":num_legs,"duty_min":duty_min,"block_min":block_min,"deadhead_min":dh_out+dh_back,
        "overnight_station":overnight_station,"hotel_nights":hotel_nights,"hotel_key":hotel_key,"hotel_day":hotel_day,
        "preference_penalty":pref_penalty,"expected_delay_risk":round(expected_delay_risk,2),
        "cost":round(cost,2), **{f"risk_{s}": round(v,2) for s, v in scenario_exposure.items()}
    }

pairings = []
legs = []
seen = set()
pid = 1

# Single flight recovery/legal fallback pairings from compatible bases.
for f in flights.itertuples(index=False):
    for base in BASES:
        # choose only bases with plausible deadhead; all base combos are possible but expensive
        for qual in ([f.qual_required, "BOTH"] if f.qual_required != "BOTH" else ["BOTH"]):
            seq = flights[flights.flight_id == f.flight_id]
            key = (base, qual, tuple(seq.flight_id), "recovery_single")
            if key in seen:
                continue
            seen.add(key)
            row = pairing_cost(base, qual, seq, "recovery_single")
            row.update({"pairing_id":f"P{pid:05d}", "pairing_type":"recovery_single"})
            pairings.append(row)
            legs.append({"pairing_id":f"P{pid:05d}", "flight_id":f.flight_id, "leg_position":1})
            pid += 1

# Generate natural round-trip and multi-leg pairings.
for base in BASES:
    candidate_starts = flights[flights.origin == base]
    for f1 in candidate_starts.itertuples(index=False):
        # 2-leg same-day returns
        nexts = flights[(flights.origin == f1.destination) & (flights.departure_min >= f1.arrival_min + MIN_CONN) & (flights.departure_min <= f1.arrival_min + MAX_SIT)]
        for f2 in nexts.itertuples(index=False):
            seq_ids = [f1.flight_id, f2.flight_id]
            seq = flights[flights.flight_id.isin(seq_ids)].sort_values("departure_min")
            ok, _ = legal_sequence(seq)
            if not ok:
                continue
            for qual in ["NB","RJ","BOTH"]:
                if not compatible_qual(qual, seq):
                    continue
                key = (base, qual, tuple(seq.flight_id), "round_trip")
                if key in seen:
                    continue
                seen.add(key)
                row = pairing_cost(base, qual, seq, "round_trip")
                row.update({"pairing_id":f"P{pid:05d}", "pairing_type":"round_trip"})
                pairings.append(row)
                for pos, fid in enumerate(seq.flight_id, 1):
                    legs.append({"pairing_id":f"P{pid:05d}", "flight_id":fid, "leg_position":pos})
                pid += 1
        # Overnight outbound + next-day inbound from same spoke
        nextday = flights[(flights.origin == f1.destination) & (flights.day == int(f1.day)+1) & (flights.departure_min >= f1.arrival_min + MIN_REST)]
        for f2 in nextday.head(3).itertuples(index=False):
            seq = flights[flights.flight_id.isin([f1.flight_id, f2.flight_id])].sort_values("departure_min")
            ok, _ = legal_sequence(seq)
            if not ok:
                continue
            for qual in ["NB","RJ","BOTH"]:
                if not compatible_qual(qual, seq):
                    continue
                key = (base, qual, tuple(seq.flight_id), "overnight")
                if key in seen:
                    continue
                seen.add(key)
                row = pairing_cost(base, qual, seq, "overnight")
                row.update({"pairing_id":f"P{pid:05d}", "pairing_type":"overnight"})
                pairings.append(row)
                for pos, fid in enumerate(seq.flight_id, 1):
                    legs.append({"pairing_id":f"P{pid:05d}", "flight_id":fid, "leg_position":pos})
                pid += 1

# Add selected 3-leg rotations from base -> spoke -> base -> spoke where legal.
for base in BASES:
    fs = flights.sort_values("departure_min")
    for f1 in fs[fs.origin == base].itertuples(index=False):
        f2s = fs[(fs.origin == f1.destination) & (fs.departure_min >= f1.arrival_min + MIN_CONN) & (fs.departure_min <= f1.arrival_min + MAX_SIT)]
        for f2 in f2s.head(2).itertuples(index=False):
            f3s = fs[(fs.origin == base) & (fs.departure_min >= f2.arrival_min + MIN_CONN) & (fs.departure_min <= f2.arrival_min + MAX_SIT)]
            for f3 in f3s.head(2).itertuples(index=False):
                seq = flights[flights.flight_id.isin([f1.flight_id, f2.flight_id, f3.flight_id])].sort_values("departure_min")
                ok, _ = legal_sequence(seq)
                if not ok:
                    continue
                for qual in ["NB","RJ","BOTH"]:
                    if not compatible_qual(qual, seq):
                        continue
                    key = (base, qual, tuple(seq.flight_id), "three_leg")
                    if key in seen:
                        continue
                    seen.add(key)
                    row = pairing_cost(base, qual, seq, "three_leg")
                    row.update({"pairing_id":f"P{pid:05d}", "pairing_type":"three_leg"})
                    pairings.append(row)
                    for pos, fid in enumerate(seq.flight_id, 1):
                        legs.append({"pairing_id":f"P{pid:05d}", "flight_id":fid, "leg_position":pos})
                    pid += 1

pairings_df = pd.DataFrame(pairings)
legs_df = pd.DataFrame(legs)
# Keep a rich but not huge master. Always keep all recovery columns, top alternatives by cost.
pairings_df = pairings_df.sort_values(["pairing_type", "cost"]).reset_index(drop=True)
# Reassign ids after filtering not necessary; preserve ids.
pairings_df.to_csv(PROC / "candidate_pairings.csv", index=False)
legs_df[legs_df.pairing_id.isin(set(pairings_df.pairing_id))].to_csv(PROC / "pairing_legs.csv", index=False)
print(f"Generated {len(pairings_df)} candidate pairings and {len(legs_df)} pairing-leg rows.")
