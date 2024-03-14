from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID, Action
import numpy as np
import json


tables = [{
    "score": [5, 0],
    "all_scores": [17, 3],
    "bishopstable": [
      -216.75723457336426, 249.06839752197266, 477.13231086730957, 383.4157485961914, -157.4692783355713,
      -232.68791389465332, 200.75105667114258, 345.85084342956543, -33.2839412689209, -106.03567123413086,
      -39.30106544494629, 98.43655586242676, -576.9510459899902, -303.10807037353516, -59.042850494384766,
      150.0783405303955, -135.3776912689209, -16.086021423339844, -102.88996505737305, -241.69321060180664,
      620.6163101196289, 320.88014030456543, 172.9753932952881, 152.30856895446777, 302.88180923461914
    ],
    "knightstable": [
      -340.6204128265381, 674.7871990203857, -537.2700786590576, -470.2384147644043, 66.02110290527344,
      -828.8660259246826, 954.701000213623, -390.301233291626, 747.3473873138428, -398.1759605407715,
      229.38384437561035, -228.95374488830566, -289.80366134643555, -62.97134590148926, -204.71296310424805,
      234.0615749359131, -562.507007598877, 77.87088012695312, 15.336227416992188, -153.08904647827148,
      -33.803489685058594, 108.22078132629395, -163.17919158935547, 205.52254676818848, 452.62900733947754
    ],
    "queenstable": [
      300.2312755584717, -121.30952072143555, -145.7700366973877, 100.48470497131348, -285.85529136657715,
      -528.9943466186523, -245.65678787231445, 226.55421447753906, 156.1007957458496, 308.2570114135742,
      316.4291744232178, 159.4451446533203, 258.99142265319824, 465.91160011291504, 257.5660285949707,
      27.760318756103516, -116.2500171661377, -105.87557029724121, -251.17243766784668, -301.49441146850586,
      208.88247680664062, -264.6044006347656, -222.99322509765625, -49.74905586242676, 672.4288902282715
    ],
    "kingstable": [
      41.05270576477051, -377.9722442626953, -47.19857978820801, 378.0010070800781, 550.7031707763672,
      52.89259719848633, -587.5678081512451, -241.9254207611084, -751.6478614807129, 59.76600646972656,
      -338.1867218017578, 104.48945999145508, 55.3389186859131, 112.81287002563477, -17.086971282958984,
      -96.3342342376709, 177.86069679260254, -39.013057708740234, -160.3330249786377, -223.95656394958496,
      296.2237033843994, -759.1101627349854, 282.42049980163574, -196.86219024658203, 86.4445571899414
    ],
    "bishopweight": 815.1852226257324,
    "knightweight": 555.9068088531494,
    "queenweight": 1347.7687454223633,
    "kingweight": 114.54794692993164,
    "knight_attacking_value": [-210.14238929748535, 304.9807777404785, -0.9197940826416016],
    "black_knight_attacking_value": [-586.4434356689453, 14.853343963623047, -175.9599151611328],
    "bishop_attacking_value": [-646.1676158905029, -79.62968444824219, 99.28949737548828],
    "black_bishop_attacking_value": [223.80562019348145, -166.52964401245117, 174.70190238952637],
    "queen_attacking_value": [-247.5411777496338, -16.178770065307617, 270.4558334350586],
    "black_queen_attacking_value": [40.173583984375, -5.511262893676758, 17.36613655090332],
    "knight_pin_value": 257.6691703796387,
    "bishop_pin_value": 277.67792320251465,
    "queen_pin_value": 452.2002143859863
  }]

class TestingAgent():
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.__player = None
        self.side = None
        
        with open('tables.json', 'r') as f:
            tables2 = json.load(f)
        
        self.knightweight = tables[-1]['knightweight']
        self.bishopweight = tables[-1]['bishopweight']
        self.queenweight = tables[-1]['queenweight']
        self.kingweight = tables[-1]['kingweight']
        
        self.knight_attacking_value = tables[-1]['knight_attacking_value']
        self.black_knight_attacking_value = tables[-1]['black_knight_attacking_value']
        self.bishop_attacking_value = tables[-1]['bishop_attacking_value']
        self.black_bishop_attacking_value = tables[-1]['black_bishop_attacking_value']
        self.queen_attacking_value = tables[-1]['queen_attacking_value']
        self.black_queen_attacking_value = tables[-1]['black_queen_attacking_value']
        
        self.knight_pinned_value = tables[-1]['knight_pin_value']
        self.bishop_pinned_value = tables[-1]['bishop_pin_value']
        self.queen_pinned_value = tables[-1]['queen_pin_value']
        
        self.bishopstable = self.reverse_table_reshape(tables[-1]['bishopstable'])
        self.knightstable = self.reverse_table_reshape(tables[-1]['knightstable'])
        self.queenstable = self.reverse_table_reshape(tables[-1]['queenstable'])
        self.kingstable = self.reverse_table_reshape(tables[-1]['kingstable'])
        
        self.mobility_score = 0.1

    def reverse_table_reshape(self, table):
        board_2d = [table[i:i+5] for i in range(0, len(table), 5)]
        original_board_2d = [row + [0]*3 for row in board_2d] + [[0]*8]*3
        original_board = [cell for row in original_board_2d for cell in row]
        return original_board

    @staticmethod
    def info():
        return {
            "agent name": "Testing agent",
        }
    
    def order_moves(self, moves, state):
        # Prioritize moves based on a simple heuristic: captures, then checks
        captures = [move for move in moves if state.board.is_capture(move.chessmove)]
        checks = [move for move in moves if state.board.gives_check(move.chessmove)]
        others = [move for move in moves if move not in captures and move not in checks]
        return captures + checks + others
    
    def alphabeta(self, alpha, beta, depthleft, state):
        if depthleft == 0:
            return self.quiesce(alpha, beta, state)
        bestscore = -9999
        moves = self.order_moves(state.applicable_moves(), state)  # Order moves
        for move in moves:
            state.execute_move(move)
            score = -self.alphabeta(-beta, -alpha, depthleft - 1, state)
            state.undo_last_move()
            if score >= beta:
                return score
            if score > bestscore:
                bestscore = score
            if score > alpha:
                alpha = score
        return bestscore
    
    def quiesce(self, alpha, beta, state: State):
        stand_pat = self.custom_evaluate_board(state)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in state.applicable_moves():
            if state.board.is_capture(move.chessmove):
                state.execute_move(move)
                score = -self.quiesce(-beta, -alpha, state)
                state.undo_last_move()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

        return alpha
    
    def custom_evaluate_board(self, state: State):
        my_turn = state.board.turn
        id = state.current_player()
        is_white = id == 0
        if state.board.is_checkmate():
            return 9999
        #There doesn't seem to be other ways to draw in this variation of chess but to repeat same position
        if state.board.is_insufficient_material():
            return -9999
        if state.board.is_stalemate():
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
                        
        def mobility_evaluation(state: State, color):
            legal_moves = list(state.board.legal_moves)
            mobility = sum(1 for move in legal_moves if state.board.piece_at(move.from_square).color == color)
            return mobility
        
        white_mobility = mobility_evaluation(state, chess.WHITE)
        black_mobility = mobility_evaluation(state, chess.BLACK)
        mobility_score = (white_mobility - black_mobility)
        
        eval = material + knight_eval + bishop_eval + queens_eval + kings_eval + pinned_val * 0.1 + attacking_val * 0.1 + mobility_score * self.mobility_score
        if not is_white:
            eval = -eval

        return eval

    def decide(self, state: AbstractState):
        if state.current_player() == 0 and state.board.fullmove_number == 1:
            # First move as white
            chessmove = chess.Move.from_uci("b1c2")
            #chessmove = chess.Move.from_uci("a1b3")
            #chessmove = chess.Move.from_uci("d1b3")
            action = Action(chessmove)
            self.side = "white"
            yield action
            return
        if state.current_player() == 1 and state.board.fullmove_number == 1:
            self.side = "black"
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
