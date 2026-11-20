"""Synthetic tabletop used by the grasp loop."""

from __future__ import annotations

import random
from dataclasses import dataclass


COLORS = ("red", "green", "blue", "yellow")
TABLE_MIN = 0.08
TABLE_MAX = 0.92
MIN_SEPARATION = 0.22


@dataclass(frozen=True)
class Block:
    name: str
    color: str
    x: float
    y: float
    radius: float = 0.06


@dataclass(frozen=True)
class Scene:
    blocks: tuple[Block, ...]
    gripper: tuple[float, float]

    def get(self, name: str) -> Block:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)


def _separated(points: list[tuple[float, float]], point: tuple[float, float]) -> bool:
    return all(
        (point[0] - x) ** 2 + (point[1] - y) ** 2 >= MIN_SEPARATION**2
        for x, y in points
    )


def make_scene(rng: random.Random, count: int | None = None) -> Scene:
    """Place colored blocks so a straight grasp path has a defined target."""
    count = count if count is not None else rng.randint(3, 4)
    points: list[tuple[float, float]] = []
    blocks: list[Block] = []
    colors = list(COLORS)
    rng.shuffle(colors)
    for index in range(count):
        placed = False
        for _ in range(80):
            point = (rng.uniform(TABLE_MIN, TABLE_MAX), rng.uniform(TABLE_MIN, TABLE_MAX))
            if _separated(points, point):
                points.append(point)
                blocks.append(
                    Block(name=f"block-{index}", color=colors[index % len(colors)], x=point[0], y=point[1])
                )
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to sample a separated tabletop")
    gripper = (rng.uniform(TABLE_MIN, TABLE_MAX), 1.05)
    return Scene(blocks=tuple(blocks), gripper=gripper)
