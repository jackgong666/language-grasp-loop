import unittest

from language_grasp.control import execute
from language_grasp.scene import Block, Scene


class ControlTests(unittest.TestCase):
    def test_reaches_an_open_target(self) -> None:
        scene = Scene(
            blocks=(Block("block-0", "red", 0.3, 0.4), Block("block-1", "blue", 0.8, 0.8)),
            gripper=(0.3, 1.05),
        )
        ok, pose = execute(scene, scene.blocks[0])
        self.assertTrue(ok)
        self.assertAlmostEqual(pose[0], 0.3, places=2)

    def test_sidestep_still_reaches_when_another_block_is_nearby(self) -> None:
        scene = Scene(
            blocks=(
                Block("block-0", "red", 0.55, 0.35),
                Block("block-1", "blue", 0.55, 0.62),
            ),
            gripper=(0.55, 1.05),
        )
        ok, _pose = execute(scene, scene.blocks[0])
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
