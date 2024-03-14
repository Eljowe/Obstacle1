from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID, Action
import numpy as np
import json


tables = [{
    "score": [5, 0],
    "all_scores": [15, 5],
    "bishopstable": [
      50.69140625, -45.23365783691406, 29.441375732421875, 61.50444793701172, -31.549747467041016, 11.36370849609375,
      -8.747581481933594, -54.407203674316406, -37.67803764343262, 77.02832412719727, -17.692920684814453,
      -93.6828556060791, 89.52383804321289, 2.2753067016601562, -3.8529205322265625, -50.394596099853516,
      15.784378051757812, -76.33856582641602, -85.55720520019531, 69.52471542358398, -110.97022438049316,
      -31.66977310180664, 4.374805450439453, 30.426177978515625, -64.51810073852539
    ],
    "knightstable": [
      -72.52500915527344, 79.96833419799805, 93.97004699707031, -62.18028450012207, -55.150352478027344,
      -72.00604248046875, 115.32371520996094, 52.74340057373047, -63.59907913208008, 15.69808578491211,
      36.257816314697266, -36.80743408203125, 43.687259674072266, -42.02659606933594, 53.51271057128906,
      -47.97089195251465, -16.38626480102539, -33.05472946166992, -8.51275634765625, -35.534793853759766,
      -35.598819732666016, 39.661258697509766, 4.197513580322266, 6.977970123291016, -73.39504623413086
    ],
    "queenstable": [
      2.914592742919922, 40.68915939331055, 78.24488067626953, -44.32709503173828, 78.53040313720703, 53.69081115722656,
      104.52727890014648, 49.66327667236328, 31.033634185791016, 12.172399520874023, -69.87245178222656,
      -64.9787483215332, 1.4916133880615234, 26.97245216369629, 78.3995246887207, -6.937870025634766, 84.65779113769531,
      -83.14295768737793, 5.880664825439453, 21.56527328491211, -48.07271957397461, 66.99410247802734,
      -14.456085205078125, -3.9567108154296875, -85.50821685791016
    ],
    "kingstable": [
      103.91489028930664, -13.115974426269531, 14.261947631835938, -10.164356231689453, -25.043380737304688,
      -5.316387176513672, -30.8527889251709, 19.457359313964844, -73.26431274414062, 15.13931655883789,
      -74.92398834228516, -22.773311614990234, 22.797645568847656, -25.721460342407227, 0.5764541625976562,
      -36.8946418762207, 23.332366943359375, -111.95217704772949, -1.136749267578125, -62.55600547790527,
      21.44223403930664, -27.853200912475586, -56.29977607727051, -71.01137924194336, 52.10188293457031
    ],
    "bishopweight": 833.0616340637207,
    "knightweight": 543.396598815918,
    "queenweight": 1309.469654083252,
    "kingweight": 158.84273147583008,
    "knight_attacking_value": [-141.76798820495605, 304.9807777404785, -0.9197940826416016],
    "black_knight_attacking_value": [-619.6837730407715, 14.853343963623047, -175.9599151611328],
    "bishop_attacking_value": [-673.779411315918, -79.62968444824219, 99.28949737548828],
    "black_bishop_attacking_value": [265.15178871154785, -166.52964401245117, 174.70190238952637],
    "queen_attacking_value": [-232.7242374420166, -16.178770065307617, 270.4558334350586],
    "black_queen_attacking_value": [31.260337829589844, -5.511262893676758, 17.36613655090332],
    "knight_pin_value": 308.4816551208496,
    "bishop_pin_value": 275.87727546691895,
    "queen_pin_value": 470.0766258239746
  }]

class TestingAgent2():
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.__player = None
        self.side = None
        
        with open('tables.json', 'r') as f:
            tables = json.load(f)
        
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
            "agent name": "Testing agent2",
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
        id = state.current_player()
        is_white = id == 0
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
