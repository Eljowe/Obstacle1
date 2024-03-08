from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID


class AgentInterface:
    """
    The interface of an Agent

    This class defines the required methods for an agent class
    """

    @staticmethod
    def info():
        """
        Return the agent's information

        This function returns the agent's information as a dictionary variable.
        The returned dictionary should contain at least the `agent name`.

        Returns
        -------
        Dict[str, str]
        """
        return {
            "agent name": "Kumpulan_Karpov",
        }
        raise NotImplementedError

    def heuristic(self, state: State):
        id = state.current_player_id
        if id == 0:
            COLOR = chess.WHITE
            otherCOLOR = chess.BLACK
        else:
            COLOR = chess.WHITE
            otherCOLOR = chess.BLACK

        knights = state.board.pieces(chess.KNIGHT, COLOR)
        bishops = state.board.pieces(chess.BISHOP, COLOR)
        queens = state.board.pieces(chess.QUEEN, otherCOLOR)

        Oknights = state.board.pieces(chess.KNIGHT, otherCOLOR)
        Obishops = state.board.pieces(chess.BISHOP, otherCOLOR)
        Oqueens = state.board.pieces(chess.QUEEN, otherCOLOR)

        score = (
            len(knights)
            + len(bishops)
            + 5 * len(queens)
            - len(Oknights)
            - len(Obishops)
            - 5 * len(Oqueens)
        )

        return score

    def decide(self, state: AbstractState):
        """
        Generate a sequence of increasing good actions

        NOTE: You can find the possible actions from `state` by calling
              `state.successors()`, which returns a list of pairs of
              `(action, successor_state)`.

        This is a generator function; it means it should have no `return`
        statement, but it should `yield` a sequence of increasing good
        actions.

        Parameters
        ----------
        state: State
            Current state of the game

        Yields
        ------
        action
            the chosen `action` from the `state.successors()` list
        """

        raise NotImplementedError

    def __str__(self):
        return self.info()["agent name"]
