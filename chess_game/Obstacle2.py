from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np

"""
This agent is a chess agent using the minimax algorithm with alpha-beta pruning and quiescence search.
The move-deciding algorithm is a simple implementation, and has no further optimizations or depth variations.

The algorithm uses a custom evaluation function to evaluate the board state.
The evaluation function is based on the material difference and the piece-square tables.

The agent also uses weights for each piece and piece-square table to evaluate the board state.
The weights and tables are based on moderate amount of reinforcement learning training with stable baselines3,
where the goal was to optimize the evaluation function to win games against other agents, as well as itself, too.

Reinforcement learning wasn't the focus of this course, but I was interested in trying to combine the methods of this course with
reinforcement learning to see if I could improve the agent's performance after implementing the basic minimax algorithm.

The weights were trained using the SAC algorithm, and the agents played against each other for approximately 10,000 games.

The agent building process was following:
0. Understand the chess library and the game environment.
1. Implement the minimax algorithm with alpha-beta pruning and quiescence search.
2. Implement the custom evaluation function.
3. Implement the weights and piece-square tables.
4. Modify custom evaluation function by adding pinning and attacking value.
5. Train the weights using reinforcement learning.
6. Testing different weights and piece-square tables to find the best performing ones.

"""

weights = {
    "score": [4, 0],
    "all_scores": [14, 2],
    "bishopstable": [
      -4.562738418579102, -92.30306816101074, 187.05451774597168, -130.44471168518066, -165.71764945983887,
      -39.211631774902344, -190.38997077941895, 84.62891578674316, -94.56267738342285, 117.95673942565918,
      159.84178161621094, 199.22202110290527, -253.69334983825684, -249.44535446166992, 41.19441223144531,
      58.11478233337402, -300.7396068572998, -157.44011116027832, -444.5422878265381, 198.84122276306152,
      114.14603424072266, -84.1431827545166, -154.59226608276367, -245.65858268737793, 130.12943840026855
    ],
    "knightstable": [
      -113.99155044555664, 262.85272216796875, 25.8481388092041, -392.55289459228516, -65.65705871582031,
      -158.95848846435547, 84.33793067932129, 45.8987979888916, 176.21803092956543, -56.16522979736328,
      132.25568008422852, -3.552732467651367, -30.494569778442383, 340.2527141571045, -7.001832962036133,
      2.2530479431152344, -48.92072677612305, 79.36643409729004, 28.115619659423828, 3.977203369140625,
      114.43006896972656, -21.6578311920166, 194.3073501586914, 115.10051727294922, -180.06785583496094
    ],
    "queenstable": [
      403.77531814575195, 16.243715286254883, -229.4129810333252, -205.47897720336914, 117.89063835144043,
      -153.08121871948242, 5.9224090576171875, 364.58356857299805, -111.08426666259766, 85.16967582702637,
      73.2731704711914, 36.81841850280762, 201.01170349121094, 6.130775451660156, 43.50358009338379, 1.4629096984863281,
      -155.19525527954102, 170.5098361968994, -111.34062194824219, -236.92815971374512, -80.6755428314209,
      -178.00924682617188, -140.79773712158203, -29.694618225097656, -4.308429718017578
    ],
    "kingstable": [
      -13.298881530761719, -250.01238822937012, -78.87904930114746, 79.80093574523926, -7.203386306762695,
      27.140392303466797, 39.51754379272461, -54.10404586791992, 113.8066635131836, -32.21948051452637,
      220.7251262664795, 12.038530349731445, 40.12166213989258, -3.799694061279297, -179.2488307952881,
      -89.54922103881836, 124.0184497833252, -90.03984260559082, -327.337589263916, 184.95351600646973,
      108.15098762512207, -183.7121696472168, 79.34293746948242, 96.81806373596191, -60.21674728393555
    ],
    "bishopweight": 467.5299816131592,
    "knightweight": 345.5488872528076,
    "queenweight": 1023.9116230010986,
    "kingweight": 337.5494918823242,
    "knight_attacking_value": [-111.73735237121582, 304.9807777404785, -0.9197940826416016],
    "black_knight_attacking_value": [25.744380950927734, 14.853343963623047, -175.9599151611328],
    "bishop_attacking_value": [-85.30455589294434, -79.62968444824219, 99.28949737548828],
    "black_bishop_attacking_value": [150.49354934692383, -166.52964401245117, 174.70190238952637],
    "queen_attacking_value": [60.02744102478027, -16.178770065307617, 270.4558334350586],
    "black_queen_attacking_value": [4.861845016479492, -5.511262893676758, 17.36613655090332],
    "knight_pin_value": -16.53989028930664,
    "bishop_pin_value": -206.0350685119629,
    "queen_pin_value": 104.54497337341309
  }
class Agent2():
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
        
        self.knight_pinned_value = weights["knight_pin_value"]
        self.bishop_pinned_value = weights["bishop_pin_value"]
        self.queen_pinned_value = weights["queen_pin_value"]
        
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
            "agent name": "Obstacle2",
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
            yield best_action
            depth += 1

    def __str__(self):
        return self.info()["agent name"]
