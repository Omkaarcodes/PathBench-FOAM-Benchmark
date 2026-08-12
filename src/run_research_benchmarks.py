import argparse
import copy
import os
import random
import tracemalloc
from typing import Any, Dict, List, Tuple

import pandas as pd

from algorithms.algorithm_manager import AlgorithmManager
from algorithms.configuration.configuration import Configuration
from maps.map_manager import MapManager
from simulator.services.services import Services
from simulator.simulator import Simulator
from structures import Point


URBAN_PREFIX = "Urban "
FOAM_PREFIX = "Probabilistic Foam"
UNSTABLE_CLASSICAL = {"Bug1", "Bug2"}

NN_ALGORITHMS = {
    "WPN",
    "WPN-view",
    "WPN-map",
    "LSTM Bagging",
    "Map Module (CAE)",
    "View Module (Online LSTM)",
}

NON_NN_ALGORITHMS = [
    "A*",
    "SPRM",
    "RT",
    "RRT",
    "RRT*",
    "RRT-Connect",
    "Wave-front",
    "Dijkstra",
    "Potential Field",
    "Probabilistic Foam (GBPF)",
    "Probabilistic Foam (PFM)",
    "Probabilistic Foam (RBPF)",
    "Probabilistic Foam (HPF)",
]


def _canonical_algorithm_name(name: str) -> str:
    return name.strip()


def _algorithm_group(name: str) -> str:
    ml_like = {
        "WPN",
        "WPN-view",
        "WPN-map",
        "LSTM Bagging",
        "Map Module (CAE)",
        "View Module (Online LSTM)",
    }
    canon = _canonical_algorithm_name(name)
    if canon in ml_like:
        return "learning_based"
    return "classical"


def _is_foam(name: str) -> bool:
    return _canonical_algorithm_name(name).startswith(FOAM_PREFIX)


def _is_urban_map(map_name: str) -> bool:
    return map_name.startswith(URBAN_PREFIX)


def _load_map(services: Services, map_name: str):
    metadata = MapManager.builtins[map_name]
    if isinstance(metadata, str):
        return services.resources.maps_dir.load(metadata)
    return copy.deepcopy(metadata)


def _valid_positions(mp) -> List[Point]:
    pts: List[Point] = []
    for y in range(mp.size.height):
        for x in range(mp.size.width):
            p = Point(x, y)
            if mp.is_agent_valid_pos(p):
                pts.append(p)
    return pts


def _sample_start_goal(valid_pts: List[Point], rng: random.Random) -> Tuple[Point, Point]:
    if len(valid_pts) < 2:
        raise ValueError("Map does not have enough valid points for start/goal sampling")

    start = rng.choice(valid_pts)
    goal = rng.choice(valid_pts)
    guard = 0
    while goal == start and guard < 20:
        goal = rng.choice(valid_pts)
        guard += 1
    if goal == start:
        raise ValueError("Failed to sample distinct start/goal points")
    return start, goal


def _simulate_once(base_map, algorithm_meta, start: Point, goal: Point) -> Dict[str, Any]:
    map_instance = copy.deepcopy(base_map)
    map_instance.trace = []
    map_instance.agent.position = start
    map_instance.goal.position = goal

    algorithm_type, testing_type, algo_params = algorithm_meta

    config = Configuration()
    config.simulator_graphics = False
    config.simulator_initial_map = map_instance
    config.simulator_algorithm_type = algorithm_type
    config.simulator_testing_type = testing_type
    config.simulator_algorithm_parameters = algo_params

    sim = Simulator(Services(config))
    tracemalloc.start()
    result = sim.start().get_results()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["memory_kib"] = peak / 1000.0
    return result


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _run_single_combination(
    map_name: str,
    algorithm_name: str,
    trials: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    base_services = Services(Configuration())
    base_map = _load_map(base_services, map_name)
    valid_pts = _valid_positions(base_map)
    map_category = "urban" if _is_urban_map(map_name) else "non_urban"

    if len(valid_pts) < 2:
        return [
            {
                "map_name": map_name,
                "map_category": map_category,
                "trial": trial_idx,
                "algorithm": _canonical_algorithm_name(algorithm_name),
                "algorithm_group": _algorithm_group(algorithm_name),
                "is_foam": _is_foam(algorithm_name),
                "start_x": float("nan"),
                "start_y": float("nan"),
                "goal_x": float("nan"),
                "goal_y": float("nan"),
                "goal_found": False,
                "total_time": float("nan"),
                "total_distance": float("nan"),
                "total_steps": float("nan"),
                "smoothness_of_trajectory": float("nan"),
                "obstacle_clearance": float("nan"),
                "distance_to_goal": float("nan"),
                "original_distance_to_goal": float("nan"),
                "map_obstacles_percentage": float("nan"),
                "memory_kib": float("nan"),
                "error": f"Map '{map_name}' has fewer than 2 valid positions",
            }
            for trial_idx in range(1, trials + 1)
        ]

    algorithm_name = _canonical_algorithm_name(algorithm_name)
    raw_key = next(
        (k for k in AlgorithmManager.builtins.keys() if _canonical_algorithm_name(k) == algorithm_name),
        None,
    )
    if raw_key is None:
        return [
            {
                "map_name": map_name,
                "map_category": map_category,
                "trial": trial_idx,
                "algorithm": algorithm_name,
                "algorithm_group": _algorithm_group(algorithm_name),
                "is_foam": _is_foam(algorithm_name),
                "start_x": float("nan"),
                "start_y": float("nan"),
                "goal_x": float("nan"),
                "goal_y": float("nan"),
                "goal_found": False,
                "total_time": float("nan"),
                "total_distance": float("nan"),
                "total_steps": float("nan"),
                "smoothness_of_trajectory": float("nan"),
                "obstacle_clearance": float("nan"),
                "distance_to_goal": float("nan"),
                "original_distance_to_goal": float("nan"),
                "map_obstacles_percentage": float("nan"),
                "memory_kib": float("nan"),
                "error": f"Unknown algorithm: {algorithm_name!r}",
            }
            for trial_idx in range(1, trials + 1)
        ]
    algorithm_meta = AlgorithmManager.builtins[raw_key]

    rows: List[Dict[str, Any]] = []

    for trial_idx in range(1, trials + 1):
        base_row: Dict[str, Any] = {
            "map_name": map_name,
            "map_category": map_category,
            "trial": trial_idx,
            "algorithm": algorithm_name,
            "algorithm_group": _algorithm_group(algorithm_name),
            "is_foam": _is_foam(algorithm_name),
        }
        try:
            start, goal = _sample_start_goal(valid_pts, rng)
            base_row.update({"start_x": start.x, "start_y": start.y,
                             "goal_x": goal.x, "goal_y": goal.y})
            result = _simulate_once(base_map, algorithm_meta, start, goal)
            row = {
                **base_row,
                "goal_found": bool(result.get("goal_found", False)),
                "total_time": _to_float(result.get("total_time")),
                "total_distance": _to_float(result.get("total_distance")),
                "total_steps": _to_float(result.get("total_steps")),
                "smoothness_of_trajectory": _to_float(result.get("smoothness_of_trajectory")),
                "obstacle_clearance": _to_float(result.get("obstacle_clearance")),
                "distance_to_goal": _to_float(result.get("distance_to_goal")),
                "original_distance_to_goal": _to_float(result.get("original_distance_to_goal")),
                "map_obstacles_percentage": _to_float(result.get("map_obstacles_percentage")),
                "memory_kib": _to_float(result.get("memory_kib")),
                "error": "",
            }
        except Exception as ex:
            row = {
                **base_row,
                "start_x": float("nan"), "start_y": float("nan"),
                "goal_x": float("nan"), "goal_y": float("nan"),
                "goal_found": False,
                "total_time": float("nan"),
                "total_distance": float("nan"),
                "total_steps": float("nan"),
                "smoothness_of_trajectory": float("nan"),
                "obstacle_clearance": float("nan"),
                "distance_to_goal": float("nan"),
                "original_distance_to_goal": float("nan"),
                "map_obstacles_percentage": float("nan"),
                "memory_kib": float("nan"),
                "error": str(ex)[:300],
            }
        rows.append(row)

    return rows


def _safe_mean(series: pd.Series) -> float:
    if series.dropna().empty:
        return float("nan")
    return float(series.mean())


def _safe_std(series: pd.Series) -> float:
    if series.dropna().empty:
        return float("nan")
    return float(series.std(ddof=0))


def _summarize_per_map(df_raw: pd.DataFrame) -> pd.DataFrame:
    grouped = df_raw.groupby(["map_name", "map_category", "algorithm", "algorithm_group", "is_foam"], dropna=False)

    summary = grouped.apply(
        lambda g: pd.Series(
            {
                "n_trials": int(len(g)),
                "success_rate": float(g["goal_found"].mean() * 100.0),
                "mean_total_time": _safe_mean(g["total_time"]),
                "std_total_time": _safe_std(g["total_time"]),
                "mean_total_distance": _safe_mean(g["total_distance"]),
                "std_total_distance": _safe_std(g["total_distance"]),
                "mean_obstacle_clearance": _safe_mean(g["obstacle_clearance"]),
                "std_obstacle_clearance": _safe_std(g["obstacle_clearance"]),
                "mean_smoothness": _safe_mean(g["smoothness_of_trajectory"]),
                "std_smoothness": _safe_std(g["smoothness_of_trajectory"]),
                "mean_memory_kib": _safe_mean(g["memory_kib"]),
                "std_memory_kib": _safe_std(g["memory_kib"]),
            }
        )
    ).reset_index()

    return summary


def _summarize_group_comparison(df_raw: pd.DataFrame, map_category: str) -> pd.DataFrame:
    subset = df_raw[df_raw["map_category"] == map_category].copy()
    subset["foam_group"] = subset["is_foam"].map(lambda v: "foam" if bool(v) else "non_foam")

    grouped = subset.groupby(["foam_group"], dropna=False)
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "n_runs": int(len(g)),
                "success_rate": float(g["goal_found"].mean() * 100.0),
                "mean_total_time": _safe_mean(g["total_time"]),
                "mean_total_distance": _safe_mean(g["total_distance"]),
                "mean_obstacle_clearance": _safe_mean(g["obstacle_clearance"]),
                "mean_smoothness": _safe_mean(g["smoothness_of_trajectory"]),
            }
        )
    ).reset_index()

    return out


def _summarize_learning_vs_classical(df_raw: pd.DataFrame) -> pd.DataFrame:
    grouped = df_raw.groupby(["map_category", "algorithm_group"], dropna=False)
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "n_runs": int(len(g)),
                "success_rate": float(g["goal_found"].mean() * 100.0),
                "mean_total_time": _safe_mean(g["total_time"]),
                "mean_total_distance": _safe_mean(g["total_distance"]),
                "mean_obstacle_clearance": _safe_mean(g["obstacle_clearance"]),
                "mean_smoothness": _safe_mean(g["smoothness_of_trajectory"]),
            }
        )
    ).reset_index()
    return out


RAW_FILENAME = "raw_runs.csv"

CSV_COLUMNS = [
    "map_name", "map_category", "trial", "algorithm", "algorithm_group", "is_foam",
    "start_x", "start_y", "goal_x", "goal_y",
    "goal_found", "total_time", "total_distance", "total_steps",
    "smoothness_of_trajectory", "obstacle_clearance",
    "distance_to_goal", "original_distance_to_goal",
    "map_obstacles_percentage", "memory_kib", "error",
]


def _compile_summaries(raw_path: str, output_dir: str) -> None:
    if not os.path.exists(raw_path):
        print(f"No raw data found at {raw_path}. Run simulations first.")
        return
    raw = pd.read_csv(raw_path)
    print(f"Loaded {len(raw)} rows from {raw_path}")

    per_map = _summarize_per_map(raw)
    per_map_path = os.path.join(output_dir, "per_map_algorithm_summary.csv")
    per_map.to_csv(per_map_path, index=False)
    print(f"Saved: {per_map_path}")

    lvsc = _summarize_learning_vs_classical(raw)
    lvsc_path = os.path.join(output_dir, "learning_vs_classical_summary.csv")
    lvsc.to_csv(lvsc_path, index=False)
    print(f"Saved: {lvsc_path}")

    foam_urban = _summarize_group_comparison(raw, "urban")
    urban_foam_path = os.path.join(output_dir, "foam_vs_nonfoam_urban.csv")
    foam_urban.to_csv(urban_foam_path, index=False)
    print(f"Saved: {urban_foam_path}")

    foam_nonurban = _summarize_group_comparison(raw, "non_urban")
    nonurban_foam_path = os.path.join(output_dir, "foam_vs_nonfoam_nonurban.csv")
    foam_nonurban.to_csv(nonurban_foam_path, index=False)
    print(f"Saved: {nonurban_foam_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PathBench research benchmark runner.\n"
            "  Single-run mode : --map MAP --algorithm ALGO\n"
            "  Compile mode    : --compile"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--map", type=str, default=None,
                        help="Built-in map name to run (required for single-run mode).")
    parser.add_argument("--algorithm", type=str, default=None,
                        help="Algorithm name to run (required for single-run mode).")
    parser.add_argument("--compile", action="store_true",
                        help="Regenerate all summary CSVs from existing raw_runs.csv.")
    parser.add_argument("--trials-per-map", type=int, default=10,
                        help="Trials per (map, algorithm) pair (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for start/goal sampling (default: 42).")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join("data", "research_results"),
                        help="Directory where CSV files are written.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, RAW_FILENAME)

    if args.compile:
        _compile_summaries(raw_path, args.output_dir)
        return

    if args.map is None or args.algorithm is None:
        parser.error(
            "Provide both --map and --algorithm for a single-run, "
            "or --compile to regenerate summaries."
        )

    algorithm = _canonical_algorithm_name(args.algorithm)
    if algorithm in NN_ALGORITHMS:
        print(f"[SKIP] '{algorithm}' is a neural-network algorithm — not included in this study.")
        return

    print(f"[RUN] map={args.map!r}  algo={algorithm!r}  trials={args.trials_per_map}")
    rows = _run_single_combination(
        map_name=args.map,
        algorithm_name=algorithm,
        trials=args.trials_per_map,
        seed=args.seed,
    )

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    write_header = not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0
    df.to_csv(raw_path, mode="a", header=write_header, index=False)

    succeeded = sum(1 for r in rows if r.get("goal_found", False))
    print(f"[DONE] {succeeded}/{len(rows)} succeeded  →  {raw_path}")


if __name__ == "__main__":
    main()