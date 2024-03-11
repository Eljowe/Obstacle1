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
        self.knightweight = 343
        self.bishopweight = 307
        self.queenweight = 721
        self.kingweight = 124
    #Change these values with deep learning
    #Also find the best first moves with deep learning
    #Or even create an opening theory with DL
        self.bishopstable = self.reverse_table_reshape([
      36.46089172363281, 70.49065399169922, -47.1213321685791, -103.83332061767578, -13.543834686279297,
      -55.394996643066406, -9.390836715698242, -17.77472496032715, -42.51819038391113, -28.45663833618164,
      -92.09583282470703, 4.234762191772461, 22.927810668945312, -68.02103424072266, 36.760101318359375,
      -6.7100677490234375, -6.053752899169922, 114.76773071289062, -30.765085220336914, 70.1805191040039,
      -45.72042655944824, -14.360719680786133, -126.61799240112305, 130.29992294311523, -68.07983589172363
    ])

        self.knightstable = self.reverse_table_reshape([
      -115.97657203674316, -128.9822769165039, 103.01798248291016, -126.62662315368652, -97.30504989624023,
      -6.4016265869140625, 68.91158676147461, 103.42523574829102, -62.52492904663086, 63.592586517333984,
      -88.40645980834961, -86.0948600769043, 29.138839721679688, -64.52531051635742, -75.01838874816895,
      12.647087097167969, 44.81605529785156, 29.174541473388672, 54.10829162597656, 66.68862915039062,
      7.323429107666016, -56.22143363952637, -25.776315689086914, 69.6229248046875, -103.65565872192383
    ])

        self.queenstable = self.reverse_table_reshape([
      -87.96242713928223, -18.091583251953125, 33.7148551940918, 20.122745513916016, -43.92212104797363,
      -12.421173095703125, 32.04261779785156, 44.52406120300293, -3.8130130767822266, 111.03653526306152,
      -14.323278427124023, -111.11234664916992, -7.781902313232422, -64.3538761138916, -136.10218811035156,
      -42.541404724121094, -73.15857696533203, 33.616302490234375, 42.55419158935547, 66.79605102539062,
      -60.457271575927734, 25.282203674316406, 49.16367721557617, -18.998348236083984, -42.16890907287598
    ])
        
        self.kingstable = self.reverse_table_reshape([
      -39.31650161743164, -31.739261627197266, 1.1877861022949219, 78.37959289550781, 106.89838790893555,
      -100.96408653259277, -36.119110107421875, -132.98211669921875, -19.265649795532227, -56.95050811767578,
      62.52158164978027, -101.15584373474121, 127.81298828125, -185.07069778442383, -57.126461029052734,
      34.6020622253418, -64.56976699829102, -29.966487884521484, -56.904741287231445, -146.8244457244873,
      -39.07763671875, -92.03934860229492, -69.49691009521484, 55.047332763671875, 49.12864685058594
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
        return {
            "agent name": "Testing agent",
        }

    
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
