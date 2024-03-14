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
      -154.1901969909668, 79.99299812316895, 249.71079635620117, -80.94643974304199, 19.684764862060547,
      -161.2284641265869, 95.02883911132812, -29.852201461791992, 118.67868995666504, 113.12360000610352,
      -17.96821403503418, 175.71012687683105, -136.7886791229248, 105.58198165893555, 230.80975341796875,
      -330.0394878387451, -23.768779754638672, -83.04229164123535, -167.57167053222656, -147.46405029296875,
      34.06406593322754, -48.636802673339844, -101.19087982177734, 121.94363021850586, -105.56432151794434
    ],
    "knightstable": [
      -148.30449676513672, 402.63076400756836, -113.73111724853516, -170.35961532592773, 71.05285835266113,
      124.7942943572998, 233.4017791748047, 154.80689239501953, -76.27298736572266, -7.343317031860352,
      -94.76200485229492, -366.772274017334, -187.35763931274414, 24.702503204345703, 35.829246520996094,
      -0.6447830200195312, -153.86131477355957, 193.63907432556152, 133.1715545654297, -189.54827690124512,
      157.62919998168945, 72.19963073730469, 79.33498191833496, 444.6170845031738, -158.83658599853516
    ],
    "queenstable": [
      121.02630996704102, 305.25220680236816, 29.074052810668945, -86.18242263793945, 116.56810188293457,
      -355.59116554260254, -191.820219039917, -37.470956802368164, 32.84990882873535, -12.861860275268555,
      -1.4485015869140625, 200.81322479248047, 105.7624568939209, -274.30699729919434, -29.98955726623535,
      -142.98265647888184, -89.15152359008789, 57.280887603759766, 175.4392852783203, -172.30377006530762,
      -53.12616157531738, -135.6097183227539, 98.42899322509766, 145.6060733795166, 8.56895637512207
    ],
    "kingstable": [
      85.08478927612305, 139.67396926879883, 91.21145057678223, 243.37818145751953, 152.8283634185791,
      36.325300216674805, -21.60225486755371, 72.96443748474121, 268.01622009277344, -39.542171478271484,
      34.44375419616699, -192.4468593597412, 33.54291343688965, -13.072555541992188, -211.72775840759277,
      30.38881492614746, -129.9224796295166, -380.35618591308594, -349.0736598968506, 158.9134006500244,
      -21.280672073364258, -244.35324478149414, 26.730003356933594, -109.36452102661133, 10.70029067993164
    ],
    "bishopweight": 387.65430641174316,
    "knightweight": 285.1974697113037,
    "queenweight": 1074.3607654571533,
    "kingweight": 490.54599952697754,
    "knight_attacking_value": [-102.19775772094727, 200.68230819702148, -75.40661430358887],
    "black_knight_attacking_value": [-44.277896881103516, 154.0067253112793, 23.229398727416992],
    "bishop_attacking_value": [31.839488983154297, 18.292619705200195, -11.061483383178711],
    "black_bishop_attacking_value": [-19.704477310180664, -33.98246192932129, -21.426347732543945],
    "queen_attacking_value": [-10.336280822753906, 85.59332656860352, -26.703746795654297],
    "black_queen_attacking_value": [275.72711753845215, -94.14887046813965, -8.749160766601562],
    "knight_pin_value": 8.141427993774414,
    "bishop_pin_value": -299.9379463195801,
    "queen_pin_value": 67.65430641174316
  }]

class TestingAgent():
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
        if state.is_winner() == 1:
            return 9999
        if state.is_winner() == -1:
            return -9999
        if id == 0 and state.board.has_insufficient_material(chess.WHITE):
            return -9999
        if id == 0 and state.board.is_stalemate():
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
