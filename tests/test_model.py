import random
import unittest

from language_grasp.language import oracle, satisfiable_instruction
from language_grasp.model import predict, train
from language_grasp.scene import make_scene


class ModelTests(unittest.TestCase):
    def test_held_out_grounding_beats_chance(self) -> None:
        weights = train(seed=0, scenes=120, epochs=6, lr=0.35)
        rng = random.Random(9)
        correct = 0
        episodes = 40
        for _ in range(episodes):
            scene = make_scene(rng)
            instruction = satisfiable_instruction(scene, rng)
            label = oracle(scene, instruction)
            assert label is not None
            choice = predict(scene, instruction, weights)
            correct += int(choice.name == label.name)
        self.assertGreaterEqual(correct / episodes, 0.9)


if __name__ == "__main__":
    unittest.main()
