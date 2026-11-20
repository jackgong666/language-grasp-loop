"""Command line entry: train, evaluate, and write results/latest.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from language_grasp.evaluate import run, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the language-conditioned grasp loop.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest.json"),
    )
    args = parser.parse_args()
    report = run(train_seed=args.train_seed, eval_seed=args.eval_seed, episodes=args.episodes)
    write_report(args.output, report)
    print(
        f"episodes={report['episodes']} "
        f"grounding_accuracy={report['grounding_accuracy']:.1%} "
        f"grasp_success={report['grasp_success']:.1%}"
    )


if __name__ == "__main__":
    main()
