import random
import unittest

from language_grasp.language import oracle, parse, satisfiable_instruction
from language_grasp.scene import Block, Scene


class ParseTests(unittest.TestCase):
    def test_color_command(self) -> None:
        instruction = parse("Pick the red block.")
        self.assertEqual(instruction.relation, "color")
        self.assertEqual(instruction.color, "red")

    def test_spatial_command(self) -> None:
        instruction = parse("pick the block left of the blue block")
        self.assertEqual(instruction.relation, "left_of")
        self.assertEqual(instruction.anchor_color, "blue")

    def test_nearest_command(self) -> None:
        instruction = parse("pick the nearest block")
        self.assertEqual(instruction.relation, "nearest")

    def test_unknown_command_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse("wave at the robot")


class OracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = Scene(
            blocks=(
                Block("block-0", "red", 0.2, 0.5),
                Block("block-1", "blue", 0.7, 0.5),
                Block("block-2", "green", 0.85, 0.8),
            ),
            gripper=(0.5, 1.05),
        )

    def test_left_of_anchor(self) -> None:
        target = oracle(self.scene, parse("pick the block left of the blue block"))
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.name, "block-0")

    def test_color_picks_the_matching_block(self) -> None:
        target = oracle(self.scene, parse("pick the green block"))
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.color, "green")

    def test_sampled_instruction_has_a_target(self) -> None:
        rng = random.Random(4)
        instruction = satisfiable_instruction(self.scene, rng)
        self.assertIsNotNone(oracle(self.scene, instruction))


if __name__ == "__main__":
    unittest.main()
