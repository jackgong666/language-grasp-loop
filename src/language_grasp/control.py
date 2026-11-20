"""Closed-loop planar approach. Replans a sidestep when the path is blocked."""

from __future__ import annotations

import math

from language_grasp.scene import Block, Scene


GRASP_TOLERANCE = 0.05
CLEARANCE = 0.09


def execute(scene: Scene, target: Block, steps: int = 12) -> tuple[bool, tuple[float, float]]:
    """Move the gripper to the target. Returns success and the final gripper pose."""
    gx, gy = scene.gripper
    for point in _waypoints((gx, gy), (target.x, target.y), steps):
        if _blocked(point, scene, target.name):
            point = _sidestep(point, scene, target)
            if _blocked(point, scene, target.name):
                return False, (gx, gy)
        gx, gy = point
    reached = math.hypot(gx - target.x, gy - target.y) <= target.radius + GRASP_TOLERANCE
    return reached, (gx, gy)


def _waypoints(start: tuple[float, float], goal: tuple[float, float], steps: int) -> list[tuple[float, float]]:
    points = []
    for index in range(1, steps + 1):
        blend = index / steps
        points.append((start[0] + (goal[0] - start[0]) * blend, start[1] + (goal[1] - start[1]) * blend))
    return points


def _blocked(point: tuple[float, float], scene: Scene, ignore: str) -> bool:
    for block in scene.blocks:
        if block.name == ignore:
            continue
        if math.hypot(point[0] - block.x, point[1] - block.y) < block.radius + CLEARANCE:
            return True
    return False


def _sidestep(point: tuple[float, float], scene: Scene, target: Block) -> tuple[float, float]:
    nearest = min(
        (block for block in scene.blocks if block.name != target.name),
        key=lambda block: math.hypot(point[0] - block.x, point[1] - block.y),
        default=None,
    )
    if nearest is None:
        return point
    dx = point[0] - nearest.x
    dy = point[1] - nearest.y
    norm = math.hypot(dx, dy) or 1.0
    push = nearest.radius + CLEARANCE + 0.04
    return (nearest.x + dx / norm * push, nearest.y + dy / norm * push)
