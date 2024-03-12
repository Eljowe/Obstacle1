from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID, Action
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
      -1.3165817260742188, -54.32647514343262, 283.39254570007324, 54.241193771362305, -97.65041923522949,
      8.624094009399414, -289.9993305206299, 122.9853572845459, -93.2891902923584, 311.5648193359375, 52.60125732421875,
      227.08377838134766, -363.4749355316162, -182.24555587768555, 144.4368438720703, 91.57030487060547,
      -229.8640537261963, -227.41910362243652, -411.48789405822754, 129.84887313842773, 242.42215538024902,
      -95.6132755279541, -3.955259323120117, -160.0073699951172, 236.9867877960205
    ],
    "knightstable": [
      -172.28234100341797, 223.0459213256836, 67.91155052185059, -368.66541290283203, 29.470746994018555,
      -40.92670440673828, 136.0531711578369, -48.32846450805664, 197.08857345581055, 108.48383140563965,
      9.541500091552734, 86.56028175354004, -153.9066925048828, 387.97168159484863, 5.78706169128418, 87.00185012817383,
      2.3975353240966797, 11.02029800415039, -58.4148006439209, 15.364641189575195, -56.979509353637695,
      24.097270965576172, 195.05109024047852, 135.32243156433105, -178.48808670043945
    ],
    "queenstable": [
      560.842456817627, 106.32813835144043, -115.91402053833008, -238.2559757232666, 133.07611846923828,
      -71.37496185302734, 115.17683601379395, 196.32369804382324, -212.03493118286133, -134.81345748901367,
      10.938720703125, 105.55173301696777, 311.7333068847656, 135.5024299621582, 117.17972755432129,
      -22.754865646362305, -201.17102432250977, 73.6533317565918, -47.31594276428223, -31.06593894958496,
      -70.76598930358887, -98.83921432495117, -159.27891540527344, -25.46590805053711, 179.67540740966797
    ],
    "kingstable": [
      58.45306587219238, -230.2301368713379, -166.03585624694824, 113.82272529602051, 177.57596015930176,
      217.13628005981445, -120.21517562866211, -41.354148864746094, 119.98504638671875, 6.917354583740234,
      70.4172248840332, -24.053510665893555, 52.26346206665039, -28.052757263183594, 21.340669631958008,
      -124.06098556518555, 282.8760738372803, -90.60652542114258, -86.45013046264648, 38.33823013305664,
      146.09135055541992, -172.38956832885742, 209.30456352233887, -12.40064811706543, 45.93386459350586
    ],
    "bishopweight": 527.1901111602783,
    "knightweight": 336.857479095459,
    "queenweight": 1094.2414417266846,
    "kingweight": 293.5579586029053,
    "knight_attacking_value": [-199.1007900238037, 304.9807777404785, -0.9197940826416016],
    "black_knight_attacking_value": [-44.8298397064209, 14.853343963623047, -175.9599151611328],
    "bishop_attacking_value": [-131.7009334564209, -79.62968444824219, 99.28949737548828],
    "black_bishop_attacking_value": [96.94354057312012, -166.52964401245117, 174.70190238952637],
    "queen_attacking_value": [-1.8053054809570312, -16.178770065307617, 270.4558334350586],
    "black_queen_attacking_value": [-91.37489891052246, -5.511262893676758, 17.36613655090332],
    "knight_pin_value": 80.09872817993164,
    "bishop_pin_value": -307.56445121765137,
    "queen_pin_value": 164.20510292053223
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
        winning = state.board.is_checkmate() & (id == 0)
        losing = state.board.is_checkmate() & (id == 1)
        if state.is_winner() == 1 | winning:
            return 9999
        if state.is_winner() == -1 | losing:
            return -9999
        if state.board.is_stalemate():
            return 0
        if state.board.is_insufficient_material():
            return 0
        
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
        if state.current_player() == 0 and state.board.fullmove_number == 1:
            # First move as white
            chessmove = chess.Move.from_uci("d1b3")
            action = Action(chessmove)

            yield action
            return
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
            yield best_action
            depth += 1

    def __str__(self):
        return self.info()["agent name"]
