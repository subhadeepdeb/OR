# Airline Crew Scheduling MILP Project — Enhanced Version

This project is a resume/interview-ready **Airline Crew Scheduling and Pairing Optimization** model. It is designed to match the complexity level of a real infrastructure optimization project: it has multiple input tables, generated operational data, pairing-column construction, a proper MILP, and post-solve analytics.

The model is not a simple crew-to-flight assignment. It follows the airline industry structure of a **set-partitioning crew-pairing master problem**: first generate legal pairings, then choose the best combination of pairings so that every flight is covered exactly once while satisfying capacity, reserve, hotel, deadhead, fairness, and disruption-risk constraints.

---

## 1. Business Problem

An airline must schedule crews for a 3-day operating horizon. The planning team has 100 scheduled flights across 36 airports and 4 crew bases. Each flight requires a crew with the correct aircraft qualification. The airline wants to minimize total pairing cost while maintaining operational feasibility and schedule resilience.

The optimization must account for:

- Full flight coverage
- Crew-base availability
- Aircraft qualification compatibility
- Duty-hour and block-hour rules
- Minimum connection/turn time
- Overnight rest rules
- Hotel-room capacity at non-base stations
- Deadhead/repositioning limits
- Reserve crew requirements by base, day, and qualification pool
- Workload balance across crew bases
- Scenario-based disruption risk from weather and ATC delay scenarios
- Crew preference penalties for unfavorable overnight regions

---

## 2. Data Complexity

The project generates and uses the following files:

### Raw data

| File | Description |
|---|---|
| `airports.csv` | 36 stations with base flag, region, turn-time, hotel capacity, curfew, and deadhead-seat limits |
| `block_time_matrix.csv` | Airport-to-airport block-time matrix |
| `flights.csv` | 100 scheduled flights over 3 days with origin, destination, departure, arrival, equipment, qualification, priority, and demand weight |
| `crew.csv` | 125 individual crew records with base, qualification, seniority, max duty, and preference attributes |
| `crew_availability.csv` | Daily crew availability by base and qualification pool plus minimum reserve requirements |
| `preference_penalties.csv` | Penalty table by base, overnight region, and day |
| `delay_scenarios.csv` | Five disruption scenarios with flight-level arrival delay shocks and scenario probabilities |
| `duty_rules.csv` | Crew legality rules such as min connection, max duty, max block, min rest, and max pairing span |

### Processed data

| File | Description |
|---|---|
| `candidate_pairings.csv` | Generated legal pairing columns with cost, duty time, deadhead, hotel, preference, and delay-risk features |
| `pairing_legs.csv` | Mapping between each pairing and the flights covered by that pairing |

### Outputs

| File | Description |
|---|---|
| `selected_pairings.csv` | Pairings selected by the MILP |
| `selected_pairing_legs.csv` | Flight-level detail for selected pairings |
| `reserve_plan.csv` | Reserve crew plan by day/base/qualification |
| `base_workload_summary.csv` | Duty-hour, block-hour, deadhead, and risk summary by base |
| `flight_coverage_check.csv` | Confirms every flight is covered exactly once |
| `scenario_risk_summary.csv` | Selected-schedule risk exposure by disruption scenario |
| `solution_summary.csv` | Objective value, pairing counts, reserve totals, and feasibility checks |
| `airline_crew_scheduling_outputs.xlsx` | Multi-sheet Excel workbook with all major outputs |

---

## 3. Mathematical Formulation

### Sets

- \(F\): set of scheduled flights
- \(P\): set of generated legal crew pairings
- \(B\): set of crew bases
- \(D\): set of planning days
- \(Q\): set of crew qualification pools
- \(S\): set of disruption scenarios
- \(H\): set of station-night hotel capacity groups

### Parameters

- \(a_{fp}\): 1 if pairing \(p\) covers flight \(f\), 0 otherwise
- \(c_p\): total cost of pairing \(p\)
- \(A_{bdq}\): available crews at base \(b\), day \(d\), qualification pool \(q\)
- \(R^{min}_{bdq}\): minimum reserve crews required at base \(b\), day \(d\), qualification pool \(q\)
- \(h_{hp}\): 1 if pairing \(p\) uses a hotel room in station-night group \(h\)
- \(H_h\): hotel-room capacity for station-night group \(h\)
- \(m_{bdp}\): 1 if pairing \(p\) starts at base \(b\) on day \(d\)
- \(g_{bp}\): duty minutes contributed to base \(b\) by pairing \(p\)
- \(\rho_{sp}\): disruption-risk minutes contributed by pairing \(p\) under scenario \(s\)
- \(\bar{\rho}_s\): scenario risk budget
- \(T\): target workload level

### Decision Variables

- \(x_p \in \{0,1\}\): 1 if pairing \(p\) is selected
- \(r_{bdq} \in \mathbb{Z}_+\): reserve crews assigned at base \(b\), day \(d\), qualification \(q\)
- \(u_b^+, u_b^- \ge 0\): positive and negative base workload deviation
- \(z_s \ge 0\): disruption-risk slack for scenario \(s\)

### Objective Function

\[
\min \sum_{p \in P} c_p x_p
+ \sum_{b,d,q} C^R r_{bdq}
+ \sum_{b \in B} C^W(u_b^+ + u_b^-)
+ \sum_{s \in S} C^Z z_s
\]

The objective minimizes pairing cost, reserve cost, workload imbalance penalty, and disruption-risk slack penalty.

### Constraints

#### 1. Flight coverage / set partitioning

\[
\sum_{p \in P} a_{fp}x_p = 1 \quad \forall f \in F
\]

Every flight must be covered exactly once.

#### 2. Crew availability by base, day, and qualification

\[
\sum_{p \in P} m_{bdqp}x_p + r_{bdq} \le A_{bdq}
\quad \forall b,d,q
\]

Selected pairings and reserve crews cannot exceed available crews.

#### 3. Minimum reserve staffing

\[
r_{bdq} \ge R^{min}_{bdq}
\quad \forall b,d,q
\]

The schedule must maintain reserve coverage for disruption recovery.

#### 4. Hotel capacity

\[
\sum_{p \in P} h_{hp}x_p \le H_h
\quad \forall h \in H
\]

Overnight pairings cannot exceed available hotel capacity.

#### 5. Deadhead/repositioning budget

\[
\sum_{p \in P} \delta_{bdp}x_p \le \bar{D}_{bd}
\quad \forall b,d
\]

This prevents unrealistic crew repositioning.

#### 6. Workload balance by base

\[
\sum_{p \in P} g_{bp}x_p - u_b^+ \le T
\quad \forall b
\]

\[
-\sum_{p \in P} g_{bp}x_p - u_b^- \le -0.65T
\quad \forall b
\]

This discourages overloading one crew base while underusing another.

#### 7. Scenario-based disruption-risk budget

\[
\sum_{p \in P} \rho_{sp}x_p - z_s \le \bar{\rho}_s
\quad \forall s \in S
\]

This adds robustness against weather and ATC delay scenarios.

---

## 4. How to Run

From the project root:

```bash
pip install -r requirements.txt
python src/generate_airline_data.py
python src/build_pairings.py
python src/solve_crew_scheduling_milp.py
```

To rebuild the Excel report:

```bash
python src/create_excel_report.py
```

---

## 5. Current Solved Instance

The included solved instance produced:

- Flights: 100
- Airports/stations: 36
- Crew members: 125
- Candidate pairings: 982
- Selected pairings: 50
- Every flight covered exactly once: True
- Recovery/single pairings selected: 1
- Round-trip pairings selected: 47
- Overnight pairings selected: 1
- Three-leg pairings selected: 1
- Total reserve crews planned: 64
- Total deadhead hours: 4.73
- Total expected delay-risk minutes: 180.23
- Solver status: Optimal

---

## 6. Interview Explanation

A strong way to explain this project:

> I modeled airline crew scheduling as a MILP-based set-partitioning problem. Instead of assigning crews directly to flights, I first generated legal crew-pairing columns from flight schedules, station rules, duty limits, turn-time requirements, aircraft qualification requirements, overnight rest rules, hotel capacity, and delay scenarios. Then I solved a master MILP that selected the minimum-cost set of pairings while covering every flight exactly once. I added operational constraints for crew availability by base and qualification, reserve staffing, deadhead limits, workload balance, and stochastic disruption-risk exposure. This made the model closer to a real airline planning problem than a simple assignment model.

---

## 7. Why This Is a Proper MILP

The model has:

- Binary decision variables for selected crew pairings
- Integer reserve staffing variables
- Continuous fairness and risk-slack variables
- Linear objective function
- Linear equality and inequality constraints

Because of the binary and integer variables, the model is solved as a MILP using branch-and-bound/branch-and-cut methods.

## Gurobi Solver Version

A Gurobi-native solver is included at:

```bash
python src/solve_crew_scheduling_gurobi.py
```

This version uses `gurobipy` directly and writes Gurobi-specific outputs to `outputs/`, including:

- `selected_pairings_gurobi.csv`
- `selected_pairing_legs_gurobi.csv`
- `reserve_plan_gurobi.csv`
- `base_workload_summary_gurobi.csv`
- `flight_coverage_check_gurobi.csv`
- `scenario_risk_summary_gurobi.csv`
- `solution_summary_gurobi.csv`
- `airline_crew_scheduling_gurobi_outputs.xlsx`

Install requirements:

```bash
pip install -r requirements.txt
```

You need a valid Gurobi license to run the Gurobi version.
