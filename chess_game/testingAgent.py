from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np




class TestingAgent():
    """
    The interface of an Agent

    This class defines the required methods for an agent class
    """
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.__player = None
        self.side = None
        self.knightweight = 313.73838996887207
        self.bishopweight = 342.81944847106934
        self.queenweight = 839.6293869018555
        self.kingweight = 181.67726135253906
    #Change these values with deep learning
    #Also find the best first moves with deep learning
    #Or even create an opening theory with DL
        self.bishopstable = self.reverse_table_reshape([
      -96.05570983886719, -104.26198196411133, -202.04754066467285, 237.04800033569336, -72.77422332763672,
      25.998851776123047, 40.57411193847656, -21.10648536682129, 83.67050552368164, -189.7845058441162,
      26.57375144958496, -104.32049369812012, 123.96389389038086, -84.27835083007812, -72.37743377685547,
      -42.4422607421875, 69.8568229675293, 32.42860221862793, 62.31779479980469, -33.698537826538086, 18.34295654296875,
      -0.6818771362304688, 19.501054763793945, -146.93967628479004, -51.16409111022949
    ])
        
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
        
        self.knightstable = self.reverse_table_reshape([
      23.84937286376953, -80.43709564208984, 30.087068557739258, 30.237178802490234, 105.05706977844238,
      127.64322662353516, -108.2237663269043, 78.9032096862793, -15.502788543701172, -6.255537033081055,
      -31.34923553466797, -68.93327713012695, 47.481285095214844, -159.93327140808105, -31.953603744506836,
      74.4565658569336, 221.0810661315918, 67.44940757751465, -30.49431800842285, 49.67422103881836, 50.3000373840332,
      86.90712547302246, 56.639631271362305, 76.81136322021484, -37.68769836425781
    ])
        
        self.rookstable = [
            0, 0, 5, 0, 0,
            -5, 0, 0, 0, -5,
            -5, 0, 0, 0, -5,
            5, 10, 10, 10, 5,
            0, 0, 0, 0, 0
        ]
        
        self.queenstable = self.reverse_table_reshape([
      -38.827199935913086, 28.666080474853516, 71.07829666137695, 94.36518287658691, -7.752784729003906,
      -17.62604522705078, -49.59476852416992, -20.17849349975586, 115.10739135742188, -142.6907730102539,
      -90.19766616821289, -45.43741798400879, 76.27928352355957, -80.71013832092285, 110.61660385131836,
      86.43702697753906, 11.357372283935547, -19.33467674255371, -57.74140739440918, -69.51341247558594,
      58.95336151123047, 70.75830841064453, -37.336172103881836, -10.966909408569336, 100.85467720031738
    ])
        
        self.kingstable = self.reverse_table_reshape([
      57.00863265991211, -236.55141639709473, -42.80156135559082, 93.6600341796875, -42.720380783081055,
      -78.41510009765625, -134.65411376953125, -30.439056396484375, 73.76772689819336, 157.35504913330078,
      -27.48668670654297, -260.1192283630371, -234.40516662597656, -128.70037269592285, -29.42179298400879,
      -54.99757385253906, -25.52596664428711, -34.26597595214844, -235.50616264343262, 104.44641304016113,
      -49.67970657348633, -197.5057201385498, 88.45704460144043, 65.4302749633789, -26.293148040771484
    ])

    def reverse_table_reshape(self, table):
        # Convert the 1D list to a 2D list
        board_2d = [table[i:i+5] for i in range(0, len(table), 5)]

        # Expand the 2D list to an 8x8 board
        original_board_2d = [row + [0]*3 for row in board_2d] + [[0]*8]*3

        # Flatten the 2D board back to a 1D list
        original_board = [cell for row in original_board_2d for cell in row]
        return original_board

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
            "agent name": "Mini_Karpov_DL_Weighted",
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
        
        wn = len(state.board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(state.board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(state.board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(state.board.pieces(chess.BISHOP, chess.BLACK))
        wq = len(state.board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(state.board.pieces(chess.QUEEN, chess.BLACK))
        wk = len(state.board.pieces(chess.KING, chess.WHITE))
        bk = len(state.board.pieces(chess.KING, chess.BLACK))

        material = self.knightweight * (wn - bn) + self.bishopweight * (wb - bb) + self.queenweight * (wq - bq) + self.kingweight * (wk - bk)
            
        
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
        #print("best value: ", bestValue)
        #print("side: ", self.side)
        #print("best action: ", best_action)
        #print("custom evaluate: ", self.custom_evaluate_board(state))
        yield best_action

    def __str__(self):
        return self.info()["agent name"]
