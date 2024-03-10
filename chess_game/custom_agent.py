from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np




class CustomAgent:
    """
    The interface of an Agent

    This class defines the required methods for an agent class
    """
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.__player = None
        self.side = None
    #Change these values with deep learning
    #Also find the best first moves with deep learning
    #Or even create an opening theory with DL
        self.bishopstable = [
            -20, -10, -10, -10, -20, 0, 0, 0,
            -10, 5, 0, 5, -5, 0, 0, 0,
            -10, 3, 10, 3, -10, 0, 0, 0,
            -10, 0, 0, 0, -10, 0, 0, 0,
            -20, -10, -10, -10, -20 , 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        
        self.pawntable = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 5, 0, 0, 0,
            5, -5, 20, -5, 5, 0, 0, 0,
            50, 50, 50, 50, 50, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        
        self.knightstable = [
            -20, -10, -10, -10, -20, 0, 0, 0,
            -10, 0, 10, 0, -10, 0, 0, 0,
            -10, 10, 20, 10, -10, 0, 0, 0,
            -10, 0, 10, 0, -10, 0, 0, 0,
            -20, -10, -10, -10, -20, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        
        self.rookstable = [
            0, 0, 5, 0, 0,
            -5, 0, 0, 0, -5,
            -5, 0, 0, 0, -5,
            5, 10, 10, 10, 5,
            0, 0, 0, 0, 0
        ]
        
        self.queenstable = [
            -20, -10, -10, -5, -20, 0, 0, 0,
            -10, 0, 0, 0, -10, 0, 0, 0,
            -10, 5, 5, 5, -10, 0, 0, 0,
            -5, 5, 5, 5, -5, 0, 0, 0,
            -20, -10, -10, -5, -20 , 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        
        self.kingstable = [
            10, 20, 0, 20, 10, 0, 0, 0,
            -50, -45, -30, -45, -50, 0, 0, 0,
            -10, -50, -50, -50, -10, 0, 0, 0,
            -20, -50, -50, -50, -20, 0, 0, 0,
            -30, -40, -40, -40, -30, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]

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
    
    def check_capture(self, move):
        return False
    
    def alphabeta(self, alpha, beta, depthleft, state):
        if (depthleft == 0):
            return self.quiesce(alpha, beta, state)
        bestscore = -9999
        for move in state.applicable_moves():
            state.execute_move(move)
            score = -self.alphabeta(-beta, -alpha, depthleft - 1, state)
            state.undo_last_move()
            if (score >= beta):
                return score
            if (score > bestscore):
                bestscore = score
            if (score > alpha):
                alpha = score
        return bestscore
    
    def quiesce(self, alpha, beta, state: State):
        stand_pat = self.custom_evaluate_board(state)
        if (stand_pat >= beta):
            return beta
        if (alpha < stand_pat):
            alpha = stand_pat
        
        for move in state.applicable_moves():
            if state.board.is_capture(move.chessmove):
                    state.execute_move(move)
                    score = -self.quiesce(-beta, -alpha, state)
                    state.undo_last_move()
                    if (score >= beta):
                        return beta
                    if (score > alpha):
                        alpha = score
        
        return alpha
    
    def custom_evaluate_board(self, state: State):
        #id = state.current_player_id
        id = state.current_player()
        if state.is_winner() == 1:
            return 9999
        if state.is_winner() == -1:
            return -9999
        if id == 0:
            self.side = "white"
            COLOR = chess.WHITE
            otherCOLOR = chess.BLACK
        else:
            self.side = "black"
            COLOR = chess.BLACK
            otherCOLOR = chess.WHITE
        
        wn = len(state.board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(state.board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(state.board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(state.board.pieces(chess.BISHOP, chess.BLACK))
        wq = len(state.board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(state.board.pieces(chess.QUEEN, chess.BLACK))
        wk = len(state.board.pieces(chess.KING, chess.WHITE))
        bk = len(state.board.pieces(chess.KING, chess.BLACK))

        material = 320 * (wn - bn) + 330 * (wb - bb) + 900 * (wq - bq)
            
        
        knightsq = sum([self.knightstable[i] for i in state.board.pieces(chess.KNIGHT, chess.WHITE)])
        knightsq = knightsq + sum([-self.knightstable[chess.square_mirror(i)]
                                for i in state.board.pieces(chess.KNIGHT, chess.BLACK)])
        bishopsq = sum([self.bishopstable[i] for i in state.board.pieces(chess.BISHOP, chess.WHITE)])
        bishopsq = bishopsq + sum([-self.bishopstable[chess.square_mirror(i)]
                                for i in state.board.pieces(chess.BISHOP, chess.BLACK)])
        queensq = sum([self.queenstable[i] for i in state.board.pieces(chess.QUEEN, chess.WHITE)])
        queensq = queensq + sum([-self.queenstable[chess.square_mirror(i)]
                                for i in state.board.pieces(chess.QUEEN, chess.BLACK)])
        kingsq = sum([self.kingstable[i] for i in state.board.pieces(chess.KING, chess.WHITE)])
        kingsq = kingsq + sum([-self.kingstable[chess.square_mirror(i)]
                            for i in state.board.pieces(chess.KING, chess.BLACK)])
        
        eval = material + knightsq + bishopsq + queensq + kingsq
        if id == 0:
            return eval
        else:
            return -eval
    def evaluate_board(self, state: State):
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
        bestValue = -99999
        alpha = -100000
        beta = 100000
        moves = state.applicable_moves()
        random.shuffle(moves)
        best_action = moves[0]
        for action in moves:
            state.execute_move(action)
            action_value = -self.alphabeta(-beta, -alpha, self.depth - 1, state)
            
            if action_value > bestValue:
                bestValue = action_value
                best_action = action
                yield best_action
            if action_value > alpha:
                alpha = action_value
            state.undo_last_move()
        print("best value: ", bestValue)
        print("side: ", self.side)
        print("best action: ", best_action)
        print("custom evaluate: ", self.custom_evaluate_board(state))
        print("evaluate: ", self.evaluate_board(state))
        yield best_action

    def __str__(self):
        return self.info()["agent name"]
