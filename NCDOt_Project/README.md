# Completed EV Charging Infrastructure Planning Models

This package consolidates and corrects the uploaded EV charging location notebooks.

## Files

- `completed_ev_charging_models.py`: clean model library with four MILP models.
- `completed_ev_charging_models.ipynb`: notebook wrapper explaining and running the models.

## Required input files

Place these files in the same folder before running:

- `TrialDataNCDOT.xlsx`
- `Demand Calculation.xlsx`
- `Data.xlsx`
- `distance_matrice.xlsx`
- `power_substation.xlsx`
- `Max capacity of power_substations.xlsx`
- `raleigh_locations.csv`
- `nodes_locations.csv`

## Models

1. Base MCLP
2. Equity MCLP
3. Base FCLP
4. Equity FCLP

All models are proper MILP models. Some simpler versions could be considered pure binary/integer linear programs, which are a subclass of MILP. The completed realistic versions include binary/integer variables plus continuous MW power-flow variables, so MILP is the correct label.

## Solver

The code uses `gurobipy`. Install Gurobi and activate your license before solving.
