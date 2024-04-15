import random
from agent_interface import AgentInterface
from envs.game import State

import chess


class MinimaxAgent(AgentInterface):
    """
    An agent who plays Chess using the Minimax algorithm
    """

    def __init__(self, depth: int = 4):
        self.depth = depth
        self.__player = None

    def info(self):
        return {"agent name": f"Minimax-Parity"}

    # Simple heuristic that counts the number of pieces
    # on both sides, and gives the Queen a higher weight.
    # To avoid a player too carelessly exchanging pieces
    # (e.g. queen for a queen), the value of one's own
    # pieces are a bit higher than the opponent's
    # corresponding pieces.

    def heuristic(self, state: State,deciding_agent : int):
        if deciding_agent == 0:
            COLOR = chess.WHITE
            otherCOLOR = chess.BLACK
        else:
            COLOR = chess.BLACK
            otherCOLOR = chess.WHITE

        knights = state.board.pieces(chess.KNIGHT,COLOR)
        bishops = state.board.pieces(chess.BISHOP,COLOR)
        queens = state.board.pieces(chess.QUEEN,COLOR)

        Oknights = state.board.pieces(chess.KNIGHT,otherCOLOR)
        Obishops = state.board.pieces(chess.BISHOP,otherCOLOR)
        Oqueens = state.board.pieces(chess.QUEEN,otherCOLOR)

        score = 3 * len(knights) + 3 * len(bishops) + 7 * len(queens) - 2 * len(Oknights) - 2 * len(Obishops) - 5 * len(Oqueens)

        return score


    def decide(self, state0: State):
        """
        Get the value of each action by passing its successor to min_value
        function.
        """
        state = state0.clone() # Must clone if used inside Iterative Deepening
        deciding = state.current_player() # Which agent's decision this is?
        moves = state.applicable_moves()
        random.shuffle(moves)
        best_action = moves[0]
        max_value = float('-inf')
        for action in moves:
            state.execute_move(action)
            action_value = self.min_value(state, self.depth - 1,deciding)
            state.undo_last_move()
            if action_value > max_value:
                max_value = action_value
                best_action = action
        #print("Best action is " + str(best_action) + "  value " + str(max_value))
        yield best_action

    def max_value(self, state: State, depth: int, deciding : int):
        # Termination conditions
        winner = state.is_winner()
        if winner is not None:
            if winner == 1: return float('inf') # Maximizing player wins
            if winner == -1: return float('-inf') # Minimizing palyer wins
            return 0
        if depth == 0:
            return self.heuristic(state,deciding)

        # If it is not terminated
        moves = state.applicable_moves()
        value = float('-inf')
        for action in moves:
            state.execute_move(action)
            value = max(value, self.min_value(state, depth - 1,deciding))
            state.undo_last_move()
        return value

    def min_value(self, state: State, depth: int, deciding : int):
        # Termination conditions
        winner = state.is_winner()
        if winner is not None:
            if winner == 1: return float('-inf') # Minimizing player wins
            if winner == -1: return float('inf') # Maximizing player wins
            return 0
        if depth == 0:
            return self.heuristic(state,deciding)

        # If it is not terminated
        moves = state.applicable_moves()
        value = float('inf')
        for action in moves:
            state.execute_move(action)
            value = min(value, self.max_value(state, depth - 1,deciding))
            state.undo_last_move()
        return value
