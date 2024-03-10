import random
from agent_interface import AgentInterface
from envs.game import State, ID

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

    def heuristic(self, state: State):
        id = state.current_player_id
        if id == 0:
            COLOR = chess.WHITE
            otherCOLOR = chess.BLACK
        else:
            COLOR = chess.WHITE
            otherCOLOR = chess.BLACK

        knights = state.board.pieces(chess.KNIGHT,COLOR)
        bishops = state.board.pieces(chess.BISHOP,COLOR)
        queens = state.board.pieces(chess.QUEEN,otherCOLOR)

        Oknights = state.board.pieces(chess.KNIGHT,otherCOLOR)
        Obishops = state.board.pieces(chess.BISHOP,otherCOLOR)
        Oqueens = state.board.pieces(chess.QUEEN,otherCOLOR)

        score = len(knights) + len(bishops) + 5 * len(queens) - len(Oknights) - len(Obishops) - 5 * len(Oqueens)

        return score


    def decide(self, state: State):
        """
        Get the value of each action by passing its successor to min_value
        function.
        """
        moves = state.applicable_moves()
        random.shuffle(moves)
        best_action = moves[0]
        max_value = float('-inf')
        for action in moves:
            state.execute_move(action)
            action_value = self.min_value(state, self.depth - 1)
            state.undo_last_move()
            if action_value > max_value:
                max_value = action_value
                best_action = action
                yield best_action
        yield best_action

    def max_value(self, state: State, depth: int):
        """
        Get the value of each action by passing its successor to min_value
        function. Return the maximum value of successors.
        
        `max_value()` function sees the game from players's perspective, trying
        to maximize the value of next state.
        
        NOTE: when passing the successor to min_value, `depth` must be
        reduced by 1, as we go down the Minimax tree.
        
        NOTE: the player must check if it is the winner (or loser)
        of the game, in which case, a large value (or a negative value) must
        be assigned to the state. Additionally, if the game is not over yet,
        but we have `depth == 0`, then we should return the heuristic value
        of the current state.
        """

        # Termination conditions
        is_winner = state.is_winner()
        if is_winner is not None:
            return is_winner * float('inf')
        if depth == 0:
            return self.heuristic(state)

        # If it is not terminated
        moves = state.applicable_moves()
        value = float('-inf')
        for action in moves:
            state.execute_move(action)
            value = max(value, self.min_value(state, depth - 1))
            state.undo_last_move()
        return value

    def min_value(self, state: State, depth):
        """
        Get the value of each action by passing its successor to max_value
        function. Return the minimum value of successors.
        
        `min_value()` function sees the game from opponent's perspective, trying
        to minimize the value of next state.
        
        NOTE: when passing the successor to max_value, `depth` must be
        reduced by 1, as we go down the Minimax tree.

        NOTE: the opponent must check if it is the winner (or loser)
        of the game, in which case, a negative value (or a large value) must
        be assigned to the state. Additionally, if the game is not over yet,
        but we have `depth == 0`, then we should return the heuristic value
        of the current state.
        """

        # Termination conditions
        is_winner = state.is_winner()
        if is_winner is not None:
            return is_winner * float('-inf')
        if depth == 0:
            return -1 * self.heuristic(state)

        # If it is not terminated
        moves = state.applicable_moves()
        value = float('inf')
        for move in moves:
            state.execute_move(move)
            value = min(value, self.max_value(state, depth - 1))
            state.undo_last_move()
        return value
