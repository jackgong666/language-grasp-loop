"""Parse a pick instruction into a grounding query."""

from __future__ import annotations

from dataclasses import dataclass

from language_grasp.scene import Block, Scene


@dataclass(frozen=True)
class Instruction:
    """One supported pick command.

    relation is color, left_of, right_of, or nearest.
    """

    relation: str
    color: str | None = None
    anchor_color: str | None = None
    text: str = ""


def parse(text: str) -> Instruction:
    cleaned = " ".join(text.lower().replace(".", " ").split())
    if cleaned.startswith("pick "):
        cleaned = cleaned[len("pick ") :]
    if cleaned.startswith("the "):
        cleaned = cleaned[len("the ") :]

    if cleaned in {"nearest block", "closest block", "nearest", "closest"}:
        return Instruction(relation="nearest", text=text)

    for relation, marker in (("left_of", " left of "), ("right_of", " right of ")):
        if marker in f" {cleaned} ":
            left, right = f" {cleaned} ".split(marker, maxsplit=1)
            return Instruction(
                relation=relation,
                color=_color_in(left),
                anchor_color=_color_in(right),
                text=text,
            )

    color = _color_in(cleaned)
    if color is None:
        raise ValueError(f"unsupported instruction: {text}")
    return Instruction(relation="color", color=color, text=text)


def _color_in(text: str) -> str | None:
    for color in ("red", "green", "blue", "yellow"):
        if color in text.split():
            return color
    return None


def oracle(scene: Scene, instruction: Instruction) -> Block | None:
    """Deterministic target used as the training label and the success check."""
    blocks = list(scene.blocks)
    if instruction.relation == "nearest":
        return min(blocks, key=lambda block: _distance(scene.gripper, block))

    if instruction.relation == "color":
        matches = [block for block in blocks if block.color == instruction.color]
        if not matches:
            return None
        return min(matches, key=lambda block: _distance(scene.gripper, block))

    anchors = [block for block in blocks if block.color == instruction.anchor_color]
    if not anchors:
        return None
    anchor = anchors[0]
    if instruction.relation == "left_of":
        matches = [block for block in blocks if block.name != anchor.name and block.x < anchor.x]
    else:
        matches = [block for block in blocks if block.name != anchor.name and block.x > anchor.x]
    if instruction.color is not None:
        matches = [block for block in matches if block.color == instruction.color]
    if not matches:
        return None
    return min(matches, key=lambda block: _distance((anchor.x, anchor.y), block))


def satisfiable_instruction(scene: Scene, rng) -> Instruction:
    """Sample a command that has exactly one oracle target on this scene."""
    options: list[Instruction] = [Instruction(relation="nearest", text="pick the nearest block")]
    for block in scene.blocks:
        options.append(
            Instruction(relation="color", color=block.color, text=f"pick the {block.color} block")
        )
        for other in scene.blocks:
            if other.name == block.name:
                continue
            if other.x < block.x:
                options.append(
                    Instruction(
                        relation="left_of",
                        anchor_color=block.color,
                        text=f"pick the block left of the {block.color} block",
                    )
                )
            if other.x > block.x:
                options.append(
                    Instruction(
                        relation="right_of",
                        anchor_color=block.color,
                        text=f"pick the block right of the {block.color} block",
                    )
                )
    rng.shuffle(options)
    for instruction in options:
        if oracle(scene, instruction) is not None:
            return instruction
    raise RuntimeError("scene has no satisfiable instruction")


def _distance(origin: tuple[float, float], block: Block) -> float:
    return (origin[0] - block.x) ** 2 + (origin[1] - block.y) ** 2
