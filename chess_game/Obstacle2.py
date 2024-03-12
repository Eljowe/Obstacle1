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
    "all_scores": [12, 4],
    "bishopstable": [
      45.35713768005371, -235.28257369995117, 187.52618217468262, -73.57968711853027, -139.5872688293457,
      -40.336673736572266, -130.20791053771973, -3.1972179412841797, -50.576988220214844, 93.10937881469727,
      175.26687049865723, 97.24403381347656, -118.19230079650879, 18.28559112548828, 186.13946533203125,
      136.98114013671875, -134.61802864074707, -193.7705593109131, -253.51268768310547, 241.7192211151123,
      28.72904396057129, -135.55889511108398, 0.4164466857910156, -218.93110847473145, 78.88802337646484
    ],
    "knightstable": [
      146.09556007385254, 178.7277431488037, 13.526580810546875, -290.5183849334717, -26.53650665283203,
      -143.20709419250488, 72.96655654907227, 51.08778953552246, 109.80880546569824, -70.62262725830078,
      112.0140209197998, -127.23813819885254, -121.59333992004395, 115.31319427490234, -100.64558792114258,
      94.93245697021484, 10.860002517700195, 25.764375686645508, -3.109333038330078, -29.527881622314453,
      136.65715408325195, 78.56656837463379, 184.95698928833008, 80.19363021850586, -32.68310546875
    ],
    "queenstable": [
      80.46536064147949, -15.156614303588867, -147.81779861450195, 18.780107498168945, 231.31945991516113,
      -206.2298755645752, -145.5229434967041, 282.05591201782227, -73.71805000305176, -137.16243743896484,
      131.5288200378418, 66.53205490112305, 101.1169204711914, -269.43789863586426, 55.61714172363281, 42.9776496887207,
      -73.3067398071289, 79.37861824035645, -187.2996368408203, -147.10622024536133, -102.62596130371094,
      -9.524566650390625, -208.52034187316895, -122.15729713439941, -82.82403945922852
    ],
    "kingstable": [
      104.08560943603516, -118.61378288269043, -56.08829879760742, -7.797277450561523, -28.539648056030273,
      -132.01547622680664, -109.13315963745117, -61.67816162109375, 258.6552047729492, -54.26677703857422,
      41.633562088012695, 181.35244178771973, 27.01484489440918, -158.8862018585205, 36.155439376831055,
      43.869911193847656, 91.21776580810547, -122.76899719238281, -368.58939361572266, 115.48870849609375,
      88.26448822021484, -359.3748302459717, -33.34906768798828, 262.19990730285645, -111.47996711730957
    ],
    "bishopweight": 302.24813652038574,
    "knightweight": 345.7535648345947,
    "queenweight": 1065.888994216919,
    "kingweight": 197.1472682952881,
    "knight_attacking_value": [168.62081909179688, 304.9807777404785, -0.9197940826416016],
    "black_knight_attacking_value": [-8.622989654541016, 14.853343963623047, -175.9599151611328],
    "bishop_attacking_value": [-27.966995239257812, -79.62968444824219, 99.28949737548828],
    "black_bishop_attacking_value": [-12.442169189453125, -166.52964401245117, 174.70190238952637],
    "queen_attacking_value": [127.97771835327148, -16.178770065307617, 270.4558334350586],
    "black_queen_attacking_value": [8.466289520263672, -5.511262893676758, 17.36613655090332],
    "knight_pin_value": 122.45963859558105,
    "bishop_pin_value": -140.26153182983398,
    "queen_pin_value": -60.73687171936035
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
