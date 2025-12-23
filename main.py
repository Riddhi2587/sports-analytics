import argparse
import importlib
import json
import os
from pathlib import Path

# Map (sport, task) -> module path and human label
# You can add more tasks per sport as you expand.
PIPELINE_REGISTRY = {
    ("basketball", "count_goals"): {
        "module": "basketball.pipeline",
        "description": "Count made baskets and per-team scores",
    },
    ("basketball", "track_players"): {
        "module": "basketball.pipeline",
        "description": "Track players and ball trajectories",
    },
    ("soccer", "count_goals"): {
        "module": "soccer.pipeline",
        "description": "Count football goals and per-team scores",
    },
    ("soccer", "track_players"): {
        "module": "soccer.pipeline",
        "description": "Track players and ball on the pitch",
    },
    ("pullups", "count_reps"): {
        "module": "pullups.pipeline",
        "description": "Count pull-up repetitions per athlete",
    },
    ("tennis", "track_players"): {
        "module": "tennis.pipeline",
        "description": "Track tennis players and basic stats",
    },
}


def list_supported():
    lines = []
    for (sport, task), cfg in sorted(PIPELINE_REGISTRY.items()):
        lines.append(f"{sport}:{task} -> {cfg['description']}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic sports video analytics router"
    )
    parser.add_argument(
        "--sport",
        required=True,
        choices=sorted({s for (s, _) in PIPELINE_REGISTRY.keys()}),
        help="Sport type (e.g., basketball, soccer, pullups, tennis)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Task for this sport (e.g., count_goals, track_players, "
            "count_reps). See --list for supported combinations."
        ),
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Path to output video file (default: ./outputs/<sport>_<task>.mp4)",
    )
    parser.add_argument(
        "--summary-json",
        required=False,
        default=None,
        help="Optional path to write JSON summary (default: ./outputs/<sport>_<task>_summary.json)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported (sport, task) combos and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Supported sport/task combinations:")
        print(list_supported())
        return

    key = (args.sport, args.task)
    if key not in PIPELINE_REGISTRY:
        raise SystemExit(
            f"Unsupported combination sport={args.sport}, task={args.task}.\n"
            f"Run with --list to see supported combinations."
        )

    cfg = PIPELINE_REGISTRY[key]
    module_path = cfg["module"]

    # Resolve default paths
    video_path = args.video
    if args.output is None:
        Path("outputs").mkdir(exist_ok=True)
        output_path = os.path.join("outputs", f"{args.sport}_{args.task}.mp4")
    else:
        output_path = args.output

    if args.summary_json is None:
        Path("outputs").mkdir(exist_ok=True)
        summary_path = os.path.join("outputs", f"{args.sport}_{args.task}_summary.json")
    else:
        summary_path = args.summary_json

    # Dynamically import the sport-specific pipeline
    try:
        sport_module = importlib.import_module(module_path)
    except ImportError as e:
        raise SystemExit(
            f"Failed to import module '{module_path}' for sport={args.sport}. "
            f"Ensure {module_path.replace('.', '/')}.py exists."
        ) from e

    if not hasattr(sport_module, "run"):
        raise SystemExit(
            f"Module '{module_path}' does not define a `run(video_path, output_path, task)` function."
        )

    # Call the sport-specific pipeline
    print(
        f"Running pipeline: sport={args.sport}, task={args.task}, "
        f"video={video_path}, output={output_path}"
    )
    summary = sport_module.run(
        video_path=video_path,
        output_path=output_path,
        task=args.task,
    )

    # Write summary JSON if returned
    if summary is None:
        summary = {}
    summary.setdefault("sport", args.sport)
    summary.setdefault("task", args.task)
    summary.setdefault("video_path", video_path)
    summary.setdefault("output_path", output_path)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote output video to: {output_path}")
    print(f"Wrote summary JSON to: {summary_path}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
