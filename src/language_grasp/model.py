"""Linear grounding model. One weight vector scores every block."""

from __future__ import annotations

import math
import random

from language_grasp.language import Instruction, oracle, satisfiable_instruction
from language_grasp.scene import Block, Scene, make_scene


FEATURE_NAMES = (
    "bias",
    "color_match",
    "color_distance",
    "left_score",
    "right_score",
    "nearest_score",
)


def features(scene: Scene, instruction: Instruction, block: Block) -> list[float]:
    """Instruction-conditioned scores. Weights are learned; the predicate is not hard-coded."""
    gx, gy = scene.gripper
    distance = math.hypot(block.x - gx, block.y - gy)
    anchor = _anchor(scene, instruction)
    color_match = 0.0
    color_distance = 0.0
    left_score = 0.0
    right_score = 0.0
    nearest_score = 0.0
    if instruction.relation == "color":
        color_match = 1.0 if block.color == instruction.color else -1.0
        color_distance = -distance if block.color == instruction.color else 0.0
    elif instruction.relation == "left_of" and anchor is not None:
        left_score = _side_score(block, anchor, left=True)
    elif instruction.relation == "right_of" and anchor is not None:
        right_score = _side_score(block, anchor, left=False)
    elif instruction.relation == "nearest":
        nearest_score = -distance
    return [1.0, color_match, color_distance, left_score, right_score, nearest_score]


def train(seed: int = 0, scenes: int = 500, epochs: int = 12, lr: float = 0.35) -> list[float]:
    rng = random.Random(seed)
    weights = [0.0] * len(FEATURE_NAMES)
    for _ in range(epochs):
        for _scene_index in range(scenes):
            scene = make_scene(rng)
            instruction = satisfiable_instruction(scene, rng)
            label = oracle(scene, instruction)
            assert label is not None
            label_index = next(i for i, block in enumerate(scene.blocks) if block.name == label.name)
            rows = [features(scene, instruction, block) for block in scene.blocks]
            _sgd(weights, rows, label_index, lr)
    return weights


def predict(scene: Scene, instruction: Instruction, weights: list[float]) -> Block:
    rows = [features(scene, instruction, block) for block in scene.blocks]
    scores = [_dot(weights, row) for row in rows]
    return scene.blocks[max(range(len(scores)), key=scores.__getitem__)]


def _side_score(block: Block, anchor: Block, left: bool) -> float:
    """Closer on the requested side ranks higher. Anchor and the other side do not."""
    if block.name == anchor.name:
        return -1.0
    on_side = block.x < anchor.x if left else block.x > anchor.x
    if not on_side:
        return -1.0
    distance = math.hypot(block.x - anchor.x, block.y - anchor.y)
    return 1.0 / (distance + 0.05)


def _anchor(scene: Scene, instruction: Instruction) -> Block | None:
    if instruction.anchor_color is None:
        return None
    for block in scene.blocks:
        if block.color == instruction.anchor_color:
            return block
    return None


def _dot(weights: list[float], row: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, row))


def _softmax(scores: list[float]) -> list[float]:
    peak = max(scores)
    shifted = [math.exp(score - peak) for score in scores]
    total = sum(shifted)
    return [value / total for value in shifted]


def _sgd(weights: list[float], rows: list[list[float]], label_index: int, lr: float) -> None:
    probs = _softmax([_dot(weights, row) for row in rows])
    for index, row in enumerate(rows):
        grad = probs[index] - (1.0 if index == label_index else 0.0)
        for feature_index, value in enumerate(row):
            weights[feature_index] -= lr * grad * value
