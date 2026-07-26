# Mathematical Formulation — Enhanced Airline Crew Scheduling MILP

This document provides the formal MILP formulation used in the project. The project follows a two-stage optimization workflow:

1. Generate feasible crew-pairing columns using duty rules and operational data.
2. Solve a set-partitioning MILP to select the best pairings.

## Pairing Generation Logic

A pairing is feasible only if it satisfies:

- chronological flight order,
- airport continuity between operated legs,
- minimum connection time,
- maximum sit time,
- maximum duty time,
- maximum block time,
- minimum overnight rest,
- aircraft qualification compatibility,
- maximum pairing span.

The pairing generation step precomputes:

- coverage vector,
- duty minutes,
- block minutes,
- deadhead minutes,
- hotel usage,
- preference penalty,
- scenario-based delay-risk exposure,
- total pairing cost.

## MILP Formulation

See the README for complete sets, parameters, variables, objective, and constraints.
