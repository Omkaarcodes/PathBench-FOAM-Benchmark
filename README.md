# Safe-Driving Vehicles: Benchmarking Probabilistic FOAM Path-Planning Algorithms

This repository contains the modified [PathBench](https://github.com/perfectly-balanced/PathBench) simulation environment, custom algorithm implementations, urban map assets, automation scripts, and analysis code used for the measuring the effectiveness of Safety-Oriented Autonomous Navigation Solutions in urban environments. To learn how to run the simulation, please check out that repository's README for detailed instructions.

The study compares traditional path-planning algorithms (A*, RRT) against four custom variants of a **Probabilistic FOAM (Free-space Obstacle Avoidance Map)** algorithm types across urban and non-urban simulated environments, evaluating them on success rate, computation time, obstacle clearance, trajectory smoothness, and memory usage.


## Repository Contents

This project builds on top of PathBench and adds the following:

```
.
├── pathbench/                      # Base PathBench framework (modified)
│   ├── algorithms/
│   │   └── foam/                   # Custom FOAM algorithm variants
│   │       ├── pfm.py              # Probabilistic Foam Method (original)
│   │       ├── gbpf.py             # Goal-Biased Probabilistic Foam
│   │       ├── rbpf.py             # Radius-Biased Probabilistic Foam
│   │       └── hpf.py              # Heuristic-Guided Probabilistic Foam
│   ├── maps/
│   │   └── urban/                  # Five added urban map assets
│   │       ├── urban_city_grid.png
│   │       ├── urban_downtown.png
│   │       ├── urban_parking_lot.png
│   │       ├── urban_city_park.png
│   │       └── urban_t_intersection.png
│   └── ...                         # Remaining PathBench core files
├── scripts/
│   └── run_all_benchmarks.ps1              # PowerShell script to automate simulation batches
├── analysis/
│   ├── analyze_results.py          # Statistical analysis (chi-squared, ANOVA, Mann-Whitney U)
│   └── plots.py                    # Generates Figures 1–3 (success rate, runtime, clearance)
├── data/
│   └── research_results/
│       └── raw_runs.csv            # Full raw output of all 4,081 trials
└── README.md
```

> **Note:** The full modified simulator, automation script, FOAM algorithm code, and raw dataset are hosted externally due to size. See [Accessing the Full Project Files](#accessing-the-full-project-files) below.

## Algorithms Tested

| Category | Algorithms |
|---|---|
| Graph-based (Traditional) | A* |
| Sampling-based (Traditional) | RRT |
| Generalized (FOAM) | PFM, GBPF, RBPF, HPF |

FOAM variants are grouped together under a single `FOAM` category for statistical comparison against the combined `Traditional` (graph + sampling) group.

| Variant | Full Name | Core Bias | Main Objective |
|---|---|---|---|
| PFM | Probabilistic Foam Method (original) | Uniform breadth-first–like bubble expansion | Cover free space, ensure safe paths |
| GBPF | Goal-Biased Probabilistic Foam | Biases expansion toward goal | Reduce computation time |
| RBPF | Radius-Biased Probabilistic Foam | Selects parent bubbles by largest radius | Maximize obstacle clearance |
| HPF | Heuristic-Guided Probabilistic Foam | A*-like heuristic guidance | Shorter paths at acceptable safety |

## Maps

Five urban maps were added to PathBench to isolate environment-driven effects, in addition to PathBench's default non-urban map set (semi-structured/randomly generated obstacle fields):

| Map Name | Category | Obstacle Density |
|---|---|---|
| Urban City Grid | urban | 46.00% |
| Urban Downtown | urban | 42.00% |
| Urban Parking Lot | urban | 16.00% |
| Urban City Park | urban | 9.75% |
| Urban T-Intersection | urban | 63.25% |

## Metrics Collected

Each trial logs the following to `raw_runs.csv`:

| Column | Type | Description |
|---|---|---|
| `Algorithm` | string | Algorithm used |
| `Map Category` | string | Environment type (urban / non-urban) |
| `Goal Found` | boolean | Whether the algorithm reached the goal |
| `Total Time` | float | Time taken to complete the run (s) |
| `Obstacle Clearance` | float | Minimum distance from obstacles along the path |
| `Smoothness` | float | Path smoothness (lack of sharp turns) |
| `Group` | string (derived) | FOAM vs. Traditional classification |

Memory usage (KiB) is also logged per trial for aggregate reporting.

## Prerequisites
- **PathBench dependencies** — install via the base PathBench `requirements.txt`
- Python analysis packages:
  ```
  pip install numpy pandas scipy matplotlib seaborn
  ```

## Setup

1. Clone or download the modified PathBench environment (see [Accessing the Full Project Files](#accessing-the-full-project-files)).
2. Install PathBench's dependencies:
   ```powershell
   cd pathbench
   pip install -r requirements.txt
   ```
3. Confirm the five urban maps are present under `pathbench/maps/urban/` and that the four FOAM algorithm files are present under `pathbench/algorithms/foam/`.

## Running the Simulations

Trials are automated through a PowerShell script that iterates over every algorithm × map combination and repeats each run to account for stochastic variability (particularly relevant for RRT).

```powershell
cd scripts
.\run_all_benchmarks.ps1
```

The script will:
1. Launch PathBench in headless/benchmark mode for each algorithm–map pair.
2. Repeat each run multiple times per map (randomized-block design — every algorithm is tested on identical maps and start–goal configurations).
3. Export a structured CSV log of all trial results to `data/research_results/raw_runs.csv`.

**Note:** Machine learning–based planners (e.g., MPNet) are intentionally excluded from the automated run, as each execution took upward of 200 seconds and would distort timing results across thousands of trials.

Because RRT-Connect crashed intermittently during data collection in the original study, expect some completion errors for that algorithm; these were accounted for when computing success rates.

## Running the Statistical Analysis

Once `raw_runs.csv` is generated (or using the provided dataset), run:

```bash
cd analysis
python analyze_results.py
```

## Summary of Findings

- **4,081 total simulations** were run across urban and non-urban environments.
- **Success rate:** No significant association between algorithm group and goal completion (χ² = 0.074, p = 0.785).
- **Runtime:** No significant main effects for map category (F = 0.327, p = 0.567) or algorithm group (F = 3.605, p = 0.0577); no significant interaction (F = 0.840, p = 0.359).
- **Obstacle clearance:** Significant difference between groups (Mann–Whitney U = 1,109,586, p < 0.001) — FOAM algorithms achieved a higher median clearance (1.810) than traditional methods (1.558).

**Conclusion:** FOAM-based algorithms do not significantly improve success rate or runtime, but they provide a statistically significant improvement in obstacle clearance — a proxy for path safety — without a corresponding cost in reliability or computation time.

## Limitations

- Simulations use a 2D, point-robot model and do not account for robot size, kinematics, actuation limits, sensor noise, or real-world power variability.
- Machine learning–based planners were excluded due to computational cost.
- PathBench's non-urban maps vary in standardization, which contributed to skew between mean and median runtimes (visible in Figure 2).

## Citation / Reference Base

This project builds on:
- Toma, A.-I. et al. (2021). *PathBench: A benchmarking platform for classical and learned path planning algorithms.* arXiv. https://doi.org/10.48550/arXiv.2105.01777
- Nascimento, L. B. P. et al. (2021). *Safe path planning algorithms for mobile robots based on probabilistic foam.* Sensors, 21(12), 4156. https://doi.org/10.3390/s21124156


This project is intended for academic/research use as part of an AP Research paper. Check PathBench's own repository for its upstream license terms before redistributing the base framework.
