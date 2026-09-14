"""Generate adaptive forecast GIFs and replay tables for all prepared datasets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASET_IDS = [
    "lapd_foothill_burglary_2011_2013_grid150m_h24h",
    "lapd_n_hollywood_burglary_2011_2013_grid150m_h24h",
    "lapd_southwest_burglary_2011_2013_grid150m_h24h",
    "lapd_foothill_vehicle_stolen_2011_2013_grid150m_h24h",
    "lapd_n_hollywood_vehicle_stolen_2011_2013_grid150m_h24h",
    "lapd_southwest_vehicle_stolen_2011_2013_grid150m_h24h",
    "chicago_district011_burglary_2011_2013_grid150m_h24h",
]


def run_id_for(dataset_id: str) -> str:
    return f"{dataset_id}_adaptive_etas_gif_2012_trial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2012-05-16")
    parser.add_argument("--end", default="2013-01-10")
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--refit-days", type=int, default=7)
    parser.add_argument("--fixed-theta", type=float, default=0.35)
    parser.add_argument("--fixed-omega", type=float, default=1.0 / 14.0)
    parser.add_argument("--theta-floor", type=float, default=0.05)
    parser.add_argument("--fade-days", type=int, default=14)
    parser.add_argument("--pop-days", type=float, default=1.5)
    parser.add_argument("--smooth-sigma", type=float, default=2.4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--dataset-id", action="append", choices=DATASET_IDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset_id or DATASET_IDS
    for dataset_id in datasets:
        run_id = run_id_for(dataset_id)
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-u",
            str(ROOT / "experiments" / "animate_adaptive_forecast.py"),
            "--dataset-id",
            dataset_id,
            "--run-id",
            run_id,
            "--model-id",
            "M4_adaptive_etas_weekly_theta_floor",
            "--start",
            args.start,
            "--end",
            args.end,
            "--history-days",
            str(args.history_days),
            "--horizon-hours",
            str(args.horizon_hours),
            "--top-k",
            str(args.top_k),
            "--refit-days",
            str(args.refit_days),
            "--fixed-theta",
            str(args.fixed_theta),
            "--fixed-omega",
            str(args.fixed_omega),
            "--theta-floor",
            str(args.theta_floor),
            "--fade-days",
            str(args.fade_days),
            "--pop-days",
            str(args.pop_days),
            "--smooth-sigma",
            str(args.smooth_sigma),
            "--frame-step",
            str(args.frame_step),
            "--duration-ms",
            str(args.duration_ms),
            "--dpi",
            str(args.dpi),
        ]
        print(f"Running {run_id}...")
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
