from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np


weights = {
    "bishopstable": [
      -0.5351104736328125, 67.75389099121094, 62.19477081298828, -13.676307678222656, -26.181903839111328,
      16.395889282226562, 4.877735137939453, -42.011268615722656, 61.084991455078125, 27.158035278320312,
      -46.75889587402344, 60.415313720703125, 82.23982238769531, 33.42598342895508, 25.408077239990234,
      38.24860382080078, -9.737430572509766, -47.529945373535156, -40.558834075927734, 29.38214874267578,
      -44.40135383605957, -86.24573516845703, -35.217567443847656, -11.712505340576172, 10.277095794677734
    ],
    "knightstable": [
      -37.11063003540039, -4.381725311279297, 63.28852844238281, -84.0914077758789, -68.78004837036133,
      24.39364242553711, -0.28163909912109375, -64.04652786254883, -1.56988525390625, -76.62065887451172,
      -93.12121200561523, -39.95299530029297, 87.60726165771484, -25.231563568115234, -15.614448547363281,
      -2.2921981811523438, -24.334980010986328, 72.73082733154297, 38.46958923339844, -26.94715118408203,
      -3.1719741821289062, -31.252777099609375, 1.8460807800292969, 52.98860549926758, 55.708763122558594
    ],
    "queenstable": [
      -8.583427429199219, 6.821062088012695, -11.72122573852539, -32.630401611328125, -19.229904174804688,
      -81.85993003845215, 32.72822570800781, 5.93634033203125, -77.30665969848633, -63.57341003417969,
      -2.045196533203125, 16.367298126220703, 60.746482849121094, -22.212121963500977, 0.5939788818359375,
      -1.984090805053711, -38.49396896362305, -27.175630569458008, -4.839565277099609, -44.837005615234375,
      -26.208810806274414, -7.59765625, -71.24874687194824, -52.27927589416504, 23.43175506591797
    ],
    "kingstable": [
      -25.064701080322266, 31.75836753845215, 67.16104125976562, 55.19562911987305, 64.38150024414062,
      -33.889347076416016, -101.50999450683594, -59.20869064331055, 6.408626556396484, 3.6911849975585938,
      21.13513946533203, -100.57134628295898, -52.0190486907959, -65.4409065246582, -67.45635795593262,
      -33.34642791748047, -20.625259399414062, -47.523916244506836, -71.99042129516602, 41.59632110595703,
      19.464889526367188, -127.29568481445312, -44.3587760925293, 12.832565307617188, -55.45540237426758
    ],
    "bishopweight": 272.9850082397461,
    "knightweight": 282.5135612487793,
    "queenweight": 957.3457183837891,
    "kingweight": 116.14559555053711,
    "knight_attacking_value": [10, 30, 90],
    "black_knight_attacking_value": [-10, -30, -90],
    "bishop_attacking_value": [10, 30, 90],
    "black_bishop_attacking_value": [-10, -30, -90],
    "queen_attacking_value": [10, 30, 90],
    "black_queen_attacking_value": [-10, -30, -90],
    "knight_pinned_value": -30,
    "bishop_pinned_value": -30,
    "queen_pinned_value": -90
  }

class TestingAgent2():
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.__player = None
        self.side = None
        
        self.knightweight = weights["knightweight"]
        self.bishopweight = weights["bishopweight"]
        self.queenweight = weights["queenweight"]
        self.kingweight = weights["kingweight"]
        
        self.knight_attacking_value = weights["knight_attacking_value"]
        self.black_knight_attacking_value = weights["black_knight_attacking_value"]
        
        self.bishop_attacking_value = weights["bishop_attacking_value"]
        self.black_bishop_attacking_value = weights["black_bishop_attacking_value"]
        
        self.queen_attacking_value = weights["queen_attacking_value"]
        self.black_queen_attacking_value = weights["black_queen_attacking_value"]
        
        self.knight_pinned_value = weights["knight_pinned_value"]
        self.bishop_pinned_value = weights["bishop_pinned_value"]
        self.queen_pinned_value = weights["queen_pinned_value"]
        
        self.bishopstable = self.reverse_table_reshape(weights["bishopstable"])
        self.knightstable = self.reverse_table_reshape(weights["knightstable"])
        self.queenstable = self.reverse_table_reshape(weights["queenstable"])
        self.kingstable = self.reverse_table_reshape(weights["kingstable"])

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
            "agent name": "Testing agent2",
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
        
        white_knight = len(state.board.pieces(chess.KNIGHT, chess.WHITE))
        black_knight = len(state.board.pieces(chess.KNIGHT, chess.BLACK))
        white_bishop = len(state.board.pieces(chess.BISHOP, chess.WHITE))
        black_bishop = len(state.board.pieces(chess.BISHOP, chess.BLACK))
        white_queen = len(state.board.pieces(chess.QUEEN, chess.WHITE))
        black_queen = len(state.board.pieces(chess.QUEEN, chess.BLACK))
        white_king = len(state.board.pieces(chess.KING, chess.WHITE))
        black_king = len(state.board.pieces(chess.KING, chess.BLACK))

        material = self.knightweight * (white_knight - black_knight) + self.bishopweight * (white_bishop - black_bishop) + self.queenweight * (white_queen - black_queen) + self.kingweight * (white_king - black_king)
            
        
        knight_eval = sum([self.knightstable[i] for i in state.board.pieces(chess.KNIGHT, chess.WHITE)])
        knight_eval = knight_eval + sum([-self.knightstable[chess.square_mirror(i)]
                                for i in state.board.pieces(chess.KNIGHT, chess.BLACK)])
        bishop_eval = sum([self.bishopstable[i] for i in state.board.pieces(chess.BISHOP, chess.WHITE)])
        bishop_eval = bishop_eval + sum([-self.bishopstable[chess.square_mirror(i)]
                                for i in state.board.pieces(chess.BISHOP, chess.BLACK)])
        queens_eval = sum([self.queenstable[i] for i in state.board.pieces(chess.QUEEN, chess.WHITE)])
        queens_eval = queens_eval + sum([-self.queenstable[chess.square_mirror(i)]
                                for i in state.board.pieces(chess.QUEEN, chess.BLACK)])
        kings_eval = sum([self.kingstable[i] for i in state.board.pieces(chess.KING, chess.WHITE)])
        kings_eval = kings_eval + sum([-self.kingstable[chess.square_mirror(i)]
                            for i in state.board.pieces(chess.KING, chess.BLACK)])
        
        def evaluate_pinned(piece_set, color, value_of_pin):
            eval = 0
            for piece in piece_set:
                if state.board.is_pinned(color, piece):
                    eval = eval + value_of_pin
            return eval
        
        pinned_val = evaluate_pinned(state.board.pieces(chess.KNIGHT, chess.WHITE), chess.WHITE, self.knight_pinned_value) + evaluate_pinned(state.board.pieces(chess.KNIGHT, chess.WHITE), chess.BLACK, -self.knight_pinned_value) +\
                        evaluate_pinned(state.board.pieces(chess.BISHOP, chess.WHITE),chess.WHITE, self.bishop_pinned_value) + evaluate_pinned(state.board.pieces(chess.BISHOP, chess.BLACK),chess.BLACK, -self.bishop_pinned_value) +\
                        evaluate_pinned(state.board.pieces(chess.QUEEN, chess.WHITE),chess.WHITE, self.queen_pinned_value) + evaluate_pinned(state.board.pieces(chess.QUEEN, chess.BLACK),chess.BLACK, -self.queen_pinned_value)
                            

        def attacking_value(pieces, attacking_pieces, attacked_pieces):
            eval = 0
            for piece in pieces:
                attacked = state.board.attacks(piece)
                for i in range(0,len(attacking_pieces)):
                    num_of_attacks_on_piece_type = len(attacked.intersection(attacking_pieces[i]))
                    eval = eval + num_of_attacks_on_piece_type * attacked_pieces[i]
            return eval

        attacking_val = attacking_value(state.board.pieces(chess.KNIGHT, chess.WHITE), [state.board.pieces(chess.KNIGHT, chess.BLACK), state.board.pieces(chess.BISHOP, chess.BLACK), state.board.pieces(chess.QUEEN, chess.BLACK)], self.knight_attacking_value) +\
                        attacking_value(state.board.pieces(chess.KNIGHT, chess.BLACK), [state.board.pieces(chess.KNIGHT, chess.WHITE), state.board.pieces(chess.BISHOP, chess.WHITE), state.board.pieces(chess.QUEEN, chess.WHITE)], self.black_knight_attacking_value) +\
                        attacking_value(state.board.pieces(chess.BISHOP, chess.WHITE), [state.board.pieces(chess.KNIGHT, chess.BLACK), state.board.pieces(chess.BISHOP, chess.BLACK), state.board.pieces(chess.QUEEN, chess.BLACK)], self.bishop_attacking_value) +\
                        attacking_value(state.board.pieces(chess.BISHOP, chess.BLACK), [state.board.pieces(chess.KNIGHT, chess.WHITE), state.board.pieces(chess.BISHOP, chess.WHITE), state.board.pieces(chess.QUEEN, chess.WHITE)], self.black_bishop_attacking_value) +\
                        attacking_value(state.board.pieces(chess.QUEEN, chess.WHITE), [state.board.pieces(chess.KNIGHT, chess.BLACK), state.board.pieces(chess.BISHOP, chess.BLACK), state.board.pieces(chess.QUEEN, chess.BLACK)], self.queen_attacking_value) +\
                        attacking_value(state.board.pieces(chess.QUEEN, chess.BLACK), [state.board.pieces(chess.KNIGHT, chess.WHITE), state.board.pieces(chess.BISHOP, chess.WHITE), state.board.pieces(chess.QUEEN, chess.WHITE)], self.black_queen_attacking_value)
                        
        
        eval = material + knight_eval + bishop_eval + queens_eval + kings_eval + pinned_val + attacking_val
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
        depth = 1
        bestValue = -99999
        alpha = -100000
        beta = 100000
        moves = state.applicable_moves()
        random.shuffle(moves)
        best_action = moves[0]
        while depth < self.max_depth + 1:
            for action in moves:
                state.execute_move(action)
                action_value = -self.alphabeta(-beta, -alpha, depth, state)
                
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
            depth += 1

    def __str__(self):
        return self.info()["agent name"]
