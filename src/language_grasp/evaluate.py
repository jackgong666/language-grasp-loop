"""Train the grounder and measure held-out grasp success."""

from __future__ import annotations

import json
import random
from pathlib import Path

from language_grasp.control import execute
from language_grasp.language import oracle, satisfiable_instruction
from language_grasp.model import FEATURE_NAMES, predict, train
from language_grasp.scene import make_scene


def run(train_seed: int = 0, eval_seed: int = 1, episodes: int = 200) -> dict:
    weights = train(seed=train_seed)
    rng = random.Random(eval_seed)
    grounded = 0
    succeeded = 0
    for _ in range(episodes):
        scene = make_scene(rng)
        instruction = satisfiable_instruction(scene, rng)
        label = oracle(scene, instruction)
        assert label is not None
        choice = predict(scene, instruction, weights)
        if choice.name == label.name:
            grounded += 1
        ok, _pose = execute(scene, choice)
        if ok and choice.name == label.name:
            succeeded += 1
    report = {
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "episodes": episodes,
        "grounding_accuracy": round(grounded / episodes, 4),
        "grasp_success": round(succeeded / episodes, 4),
        "feature_count": len(FEATURE_NAMES),
        "model": "softmax linear grounder",
        "controller": "closed-loop planar approach with one sidestep",
    }
    return report


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
