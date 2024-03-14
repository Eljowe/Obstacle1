from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID, Action
from agent_interface import AgentInterface

"""
This chess agent uses the minimax algorithm with alpha-beta pruning (negamax) and quiescence search.

The algorithm uses a custom evaluation function to evaluate the board state.
The evaluation function is based on the material difference and the piece-square tables,
with the additional try of adding the mobility score, pinning and attacking value.

The weights and tables are based on moderate amount of reinforcement learning training with stable baselines3,
where the goal was to optimize the evaluation function to win games against other agents, as well as itself, too.

Reinforcement learning wasn't the focus of this course, but I was interested in trying to combine the methods of this course with
reinforcement learning to see if I could improve the agent's performance after implementing the basic minimax algorithm with alpha beta pruning.

The weights were trained using the SAC algorithm, and the agents played against each other for approximately 10,000 games.

The agent building process was following:
0. Understand the chess library and the game environment.
1. Implement the minimax algorithm with alpha-beta pruning and quiescence search.
2. Implement the custom evaluation function.
3. Implement the weights and piece-square tables.
4. Modify custom evaluation function by adding pinning and attacking value.
5. Train the weights using reinforcement learning.
6. Testing different weights and piece-square tables to find the best performing ones.
7. Finally implement iterative deepening.
8. Adding move ordering to check captures and checks first.

Results:
With time limit 2.0 seconds, agent searches usually as deep as depths 5-8 depending on the number of possible moves,
sometimes even up to 12, while minimax can do something like 5 most of the times.
With time limit 2.0 seconds, the agent can beat the provided minimax agent in 20 games with 19-1 record.
At higher time limits, the games are more balanced, but the agent still has a slight advantage.
"""

tables = [
    {
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
  }
    ]

class Agent5(AgentInterface):
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.__player = None
        self.side = None
        
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
        
        self.transposition_table = {}
        
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
            "agent name": "Obstacle5",
        }
    
    def hash_board(self, state):
        return hash(state.board.fen())
    
    def order_moves(self, moves, state):
        # Prioritize moves based on a simple heuristic: captures, then checks
        captures = [move for move in moves if state.board.is_capture(move.chessmove)]
        checks = [move for move in moves if state.board.gives_check(move.chessmove)]
        others = [move for move in moves if move not in captures and move not in checks]
        return captures + checks + others
    
    def alphabeta(self, alpha, beta, depthleft, state):
        board_hash = self.hash_board(state)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depthleft:
                return entry['score']
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
        self.transposition_table[board_hash] = {'score': bestscore, 'depth': depthleft}
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
                        
        
        
        eval = material + knight_eval + bishop_eval + queens_eval + kings_eval + pinned_val * 0.1 + attacking_val * self.mobility_score
        if not is_white:
            eval = -eval

        return eval

    def decide(self, state: AbstractState):
        if state.current_player() == 0 and state.board.fullmove_number == 1:
            # First move as white
            #chessmove = chess.Move.from_uci("b1c2")
            chessmove = chess.Move.from_uci("a1b3")
            #chessmove = chess.Move.from_uci("d1b3")
            action = Action(chessmove)
            self.side = "white"
            yield action
            return
        if state.current_player() == 1 and state.board.fullmove_number == 1:
            self.side = "black"
        depth = 2
        bestValue = -99999
        alpha = -100000
        beta = 100000
        moves = self.order_moves(state.applicable_moves(), state)
        best_action = moves[0]
        while depth < self.max_depth + 1:
            #print(f"Obstacle1 depth: {depth}")
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
