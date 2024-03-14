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
      -3.004955291748047, 76.33552551269531, -43.678993225097656, -21.549949645996094, -43.18385314941406,
      29.587730407714844, 19.067630767822266, -10.036720275878906, -25.91907501220703, -22.097753524780273,
      -11.590614318847656, 2.788015365600586, 106.180419921875, 2.7541580200195312, 6.861358642578125,
      16.464231491088867, 34.52390670776367, 7.068946838378906, 0.8978538513183594, -4.167369842529297,
      -16.23358726501465, -39.303009033203125, 59.43785095214844, 33.134971618652344, -11.951578140258789
    ],
    "knightstable": [
      0.9935798645019531, -30.051189422607422, 55.440673828125, -42.1550407409668, -29.715328216552734,
      21.30963134765625, 9.425006866455078, 23.439151763916016, -76.15379524230957, -17.35839080810547,
      -30.042497634887695, -33.86610794067383, -20.188854217529297, 50.79231643676758, -50.49796676635742,
      67.3028564453125, 53.21715545654297, -32.3087043762207, -69.81172561645508, 10.648319244384766,
      -39.44023323059082, 18.271339416503906, -26.508556365966797, 51.11777114868164, -52.342445373535156
    ],
    "queenstable": [
      27.000831604003906, 29.178638458251953, -29.51570701599121, -5.579471588134766, -34.768056869506836,
      -30.044631958007812, 65.94981384277344, 20.029067993164062, 50.478187561035156, -1.2911758422851562,
      -51.06867218017578, 63.37074279785156, 37.752017974853516, 25.009075164794922, -53.125932693481445,
      -29.573951721191406, -35.15726089477539, 22.888046264648438, 4.919589996337891, 54.66387939453125,
      1.2532997131347656, -32.72789001464844, 0.9069404602050781, -6.72752571105957, 53.18000793457031
    ],
    "kingstable": [
      52.075523376464844, -6.554927825927734, -7.190399169921875, -63.47600173950195, 43.63125228881836,
      -42.555145263671875, -41.36888885498047, -49.60041427612305, 27.051055908203125, -54.995296478271484,
      -24.50851821899414, -44.84263229370117, -116.30418586730957, -76.12959861755371, -68.60250091552734,
      -65.46415710449219, -41.859697341918945, -77.94579315185547, 4.788795471191406, -36.589515686035156,
      -64.57928466796875, -34.1733283996582, -120.03519821166992, 43.28221130371094, 35.89203643798828
    ],
    "bishopweight": 809.700065612793,
    "knightweight": 539.1559085845947,
    "queenweight": 1408.889389038086,
    "kingweight": 75.4780502319336,
    "knight_attacking_value": [-244.17661476135254, 304.9807777404785, -0.9197940826416016],
    "black_knight_attacking_value": [-492.77305603027344, 14.853343963623047, -175.9599151611328],
    "bishop_attacking_value": [-622.6140995025635, -79.62968444824219, 99.28949737548828],
    "black_bishop_attacking_value": [219.4115390777588, -166.52964401245117, 174.70190238952637],
    "queen_attacking_value": [-245.51386642456055, -16.178770065307617, 270.4558334350586],
    "black_queen_attacking_value": [-2.9569931030273438, -5.511262893676758, 17.36613655090332],
    "knight_pin_value": 282.4470748901367,
    "bishop_pin_value": 197.1545181274414,
    "queen_pin_value": 446.7150573730469
  }]

class TestingAgent3():
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.__player = None
        self.side = None
        self.piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
        
        self.transposition_table = {}
        
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

    def hash_board(self, state):
        return hash(state.board.fen())
    
    def reverse_table_reshape(self, table):
        board_2d = [table[i:i+5] for i in range(0, len(table), 5)]
        original_board_2d = [row + [0]*3 for row in board_2d] + [[0]*8]*3
        original_board = [cell for row in original_board_2d for cell in row]
        return original_board

    @staticmethod
    def info():
        return {
            "agent name": "Testing agent3",
        }
        
    def evaluate_material(self, state):
        # Example material evaluation
        material_weights = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
        score = 0
        for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K']:
            score += len(state.board.pieces(piece_type, True)) * material_weights[piece_type]
            score -= len(state.board.pieces(piece_type, False)) * material_weights[piece_type]
        return score
    
    def order_moves(self, moves, state):
        # Simple heuristic: prioritize captures, then use a basic heuristic for non-captures
        captures = sorted(moves, key=lambda move: self.capture_priority(move, state), reverse=True)
        non_captures = [move for move in moves if not state.board.is_capture(move.chessmove)]
        return captures + non_captures
    
    def capture_priority(self, move, state):
        if state.board.is_capture(move.chessmove):
            capturing_piece = state.board.piece_at(move.chessmove.from_square)
            captured_piece = state.board.piece_at(move.chessmove.to_square)
            return self.piece_values[captured_piece.symbol().upper()] - self.piece_values[capturing_piece.symbol().upper()]
        return 0

    def piece_value(self, piece):
        # Simple piece value function; you can expand this with actual piece values
        values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100}
        return values.get(piece.symbol().upper(), 0)
    
    def alphabeta(self, alpha, beta, depthleft, state):
        board_hash = self.hash_board(state)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depthleft:
                return entry['score']
        
        if depthleft == 0:
            return self.evaluate_board(state)

        bestscore = float('-inf')
        for move in self.order_moves(state.applicable_moves(), state):
            state.execute_move(move)
            score = -self.alphabeta(-beta, -alpha, depthleft - 1, state)
            state.undo_last_move()
            if score >= beta:
                return score  # Beta cutoff
            bestscore = max(bestscore, score)
            alpha = max(alpha, score)

        self.transposition_table[board_hash] = {'score': bestscore, 'depth': depthleft}
        return bestscore
    
    def evaluate_board(self, state):
        # Simple material-based evaluation
        id = state.current_player()
        is_white = id == 0
        if state.is_winner() == 1:
            return 9999
        if state.is_winner() == -1:
            return -9999
        score = 0
        piece_values = {
            chess.PAWN: 100, 
            chess.KNIGHT: 320, 
            chess.BISHOP: 330, 
            chess.ROOK: 500, 
            chess.QUEEN: 900, 
            chess.KING: 20000
        }
        for piece_type, value in piece_values.items():
            score += len(state.board.pieces(piece_type, chess.WHITE)) * value
            score -= len(state.board.pieces(piece_type, chess.BLACK)) * value
        return score

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
        moves = self.order_moves(state.applicable_moves(), state)
        best_action = moves[0]
        while depth < self.max_depth + 1:
            #print(f"testing depth: {depth}")
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
