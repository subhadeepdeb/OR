#!/usr/bin/env bash
set -e
python src/generate_airline_data.py
python src/build_pairings.py
python src/solve_crew_scheduling_milp.py
python src/create_excel_report.py
