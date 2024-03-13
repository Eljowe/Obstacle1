import unittest
import sys
from __init__ import *
from envs.game import State

class TestTestingAgent(unittest.TestCase):

    def setUp(self):
        # Initialize the agent here
        self.agent = TestingAgent(max_depth=5)  # Adjust parameters as needed

        # Assuming State() requires a list of players, and TestingAgent can be used directly
        player1 = self.agent
        player2 = TestingAgent(max_depth=5)  # For the purpose of initializing State, adjust as necessary

        # Initialize State with a list of these player instances.
        self.initial_state = State(players=[player1, player2])

    def test_decision_making(self):
        """Test the agent's decision-making capability."""
        decision = next(self.agent.decide(self.initial_state))  # If decide() is a generator
        # Assuming decide() returns a move, validate it's a valid move for the initial state
        self.assertIn(decision, self.initial_state.applicable_moves(), "Agent made an invalid move")

    def test_evaluation_function(self):
        """Test the evaluation function's correctness."""
        # Setup a specific board state here
        test_state = self.initial_state  # Modify this to represent a specific board state
        score = self.agent.custom_evaluate_board(test_state)
        # Assuming you know the expected score for the test_state
        expected_score = 0  # Replace with the actual expected score
        self.assertEqual(score, expected_score, "Evaluation function returned an incorrect score")

if __name__ == '__main__':
    unittest.main()
