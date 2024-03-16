from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID, Action
from agent_interface import AgentInterface

"""
This chess agent uses the minimax algorithm with alpha-beta pruning (negamax), quiescence search and zobrist hashing.

The algorithm uses a custom evaluation function to evaluate the board state.
The evaluation function is based on the material difference, with the additional try of adding the mobility score, pinning and attacking value.

The weights are based on moderate amount of reinforcement learning training with stable baselines3,
where the goal was to optimize the evaluation function to win games against other agents, as well as itself, too.
But as the training was conducted with 5x5 board, the weights might not be optimal for larger boards.

Reinforcement learning wasn't the focus of this course or at least this week, but I was interested in trying to combine the methods of this course with
RL methods to see if I could improve the agent's performance after implementing the basic setup.

The weights were trained using the SAC algorithm, and the agents played against each other for approximately 10,000 games,
not really showing much improvement to be honest.

The agent building process was following:
0. Understand the chess library and the game environment.
1. Implement the minimax algorithm with alpha-beta pruning and quiescence search.
2. Implement the custom evaluation function.
3. Implement the weights.
4. Modify custom evaluation function by adding pinning and attacking value.
5. Train the weights using reinforcement learning.
6. Testing different weights to find the best performing ones.
7. Finally implement iterative deepening.
8. Adding move ordering to check captures and checks first.

Results:
With time limit 2.0 seconds, agent searches usually as deep as depths 5-8 depending on the number of possible moves, which is in similar range as minimax.
With time limit 2.0 seconds and 5x5 board, the agent is pretty equal to minimax, but with larger boards and more possible moves,
the difference is clear.
With 6x6 board and 2.0s limit, the agent is easily able to beat minimax (16-4 in 20 matches).
I suppose we'll see the final results during the tournament.
"""

tables = {
    "score": [5, 0],
    "all_scores": [15, 5],
    "bishopweight": 723.5667018890381,
    "knightweight": 636.6385307312012,
    "queenweight": 1574.5610809326172,
    "kingweight": 75.55226135253906,
    "knight_attacking_value": [-279.09769439697266, 304.9807777404785, -0.9197940826416016],
    "black_knight_attacking_value": [-653.2774238586426, 14.853343963623047, -175.9599151611328],
    "bishop_attacking_value": [-584.2226734161377, -79.62968444824219, 99.28949737548828],
    "black_bishop_attacking_value": [163.3222427368164, -166.52964401245117, 174.70190238952637],
    "queen_attacking_value": [-314.4085273742676, -16.178770065307617, 270.4558334350586],
    "black_queen_attacking_value": [83.51497650146484, -5.511262893676758, 17.36613655090332],
    "knight_pin_value": 424.9740695953369,
    "bishop_pin_value": 296.96722412109375,
    "queen_pin_value": 502.3064727783203
  }


class Agent(AgentInterface):
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.__player = None
        self.side = None
        
        self.knightweight = tables["knightweight"]
        self.bishopweight = tables["bishopweight"]
        self.queenweight = tables["queenweight"]
        self.kingweight = tables["kingweight"]
        self.rookweight = 1000
        self.pawnweight = 100
        
        self.knight_attacking_value = tables["knight_attacking_value"]
        self.black_knight_attacking_value = tables["black_knight_attacking_value"]
        self.bishop_attacking_value = tables["bishop_attacking_value"]
        self.black_bishop_attacking_value = tables["black_bishop_attacking_value"]
        self.queen_attacking_value = tables["queen_attacking_value"]
        self.black_queen_attacking_value = tables["black_queen_attacking_value"]
        
        self.transposition_table = {}
        self.zobrist_table = self.initialize_zobrist_table()
        
        self.knight_pinned_value = tables["knight_pin_value"]
        self.bishop_pinned_value = tables["bishop_pin_value"]
        self.queen_pinned_value = tables["queen_pin_value"]
        
        self.mobility_multiplier = 0.1
    
    def initialize_zobrist_table(self):
        zobrist_table = []
        for _ in range(64):
            row = []
            for _ in range(12):
                row.append(random.randint(0, 2**64 - 1))
            zobrist_table.append(row)
        return zobrist_table
    
    def hash_board(self, state):
        zobrist_hash = 0
        board = state.board
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_index = piece.piece_type - 1
                if piece.color == chess.BLACK:
                    piece_index += 6
                zobrist_hash ^= self.zobrist_table[square][piece_index]
        return zobrist_hash


    @staticmethod
    def info():
        return {
            "agent name": "Obstacle1",
        }
    
    
    def order_moves(self, moves, state):
        # Prioritize moves based on a simple heuristic: captures, then checks
        captures = [move for move in moves if state.board.is_capture(move.chessmove)]
        checks = [move for move in moves if state.board.gives_check(move.chessmove)]
        others = [move for move in moves if move not in captures and move not in checks]
        return captures + checks + others
    
    def alphabeta(self, alpha, beta, depthleft, state):
        is_winner = state.is_winner()
        if is_winner is not None:
            return is_winner * float('inf')
        board_hash = self.hash_board(state)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depthleft:
                return entry['score']
        if depthleft == 0:
            return self.quiesce(alpha, beta, state)
        
        # Null move pruning
        R = 5
        if depthleft >= R and not state.board.is_check():
            null_move = chess.Move.null()
            state.board.push(null_move)
            
            score = -self.alphabeta(-beta, -beta + 1, depthleft - R, state)
            state.undo_last_move()
            if score >= beta:
                return beta
        
        bestscore = -20000
        bestmove = None
        moves = self.order_moves(state.applicable_moves(), state)
        
        #Late move reduction
        for i, move in enumerate(moves):
            state.execute_move(move)

            if i == 0:
                score = -self.alphabeta(-beta, -alpha, depthleft - 1, state)
            else:
                reduction = 1 if depthleft >= 3 and i >= 2 else 0
                score = -self.alphabeta(-alpha - 1, -alpha, depthleft - 1 - reduction, state)

                if score > alpha:
                    score = -self.alphabeta(-beta, -alpha, depthleft - 1, state)

            state.undo_last_move()

            if score >= beta:
                self.transposition_table[board_hash] = {'score': score, 'depth': depthleft, 'bestmove': move}
                return score
            if score > bestscore:
                bestscore = score
                bestmove = move
            if score > alpha:
                alpha = score

        self.transposition_table[board_hash] = {'score': bestscore, 'depth': depthleft, 'bestmove': bestmove}
        return bestscore
    
    def quiesce(self, alpha, beta, state: State):
        stand_pat = self.custom_evaluate_board(state)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        board_hash = self.hash_board(state)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= 0:
                return entry['score']

        for move in state.applicable_moves():
            if state.board.is_capture(move.chessmove):
                state.execute_move(move)
                score = -self.quiesce(-beta, -alpha, state)
                state.undo_last_move()
                if score >= beta:
                    self.transposition_table[board_hash] = {'score': score, 'depth': 0}
                    return beta
                if score > alpha:
                    alpha = score

        self.transposition_table[board_hash] = {'score': alpha, 'depth': 0}
        return alpha
    
    
    def custom_evaluate_board(self, state: State):
        
        def evaluate_pinned(piece_set, color, value_of_pin):
            eval = 0
            for piece in piece_set:
                if state.board.is_pinned(color, piece):
                    eval = eval + value_of_pin
            return eval
        
        def attacking_value(pieces, attacking_pieces, attacked_pieces):
            eval = 0
            for piece in pieces:
                attacked = state.board.attacks(piece)
                for i in range(0,len(attacking_pieces)):
                    num_of_attacks_on_piece_type = len(attacked.intersection(attacking_pieces[i]))
                    eval = eval + num_of_attacks_on_piece_type * attacked_pieces[i]
            return eval
        
        def mobility_evaluation(state: State, color):
            legal_moves = list(state.board.legal_moves)
            mobility = sum(1 for move in legal_moves if state.board.piece_at(move.from_square).color == color)
            return mobility
        
        id = state.current_player()
        is_white = id == 0

        winner = state.is_winner()
        if winner == 1:
            if is_white:
                return 9999  # White wins
            else:
                return -9999  # Black wins
        elif winner == -1:
            if is_white:
                return -9999  # Black wins
            else:
                return 9999
            
        white_knight = len(state.board.pieces(chess.KNIGHT, chess.WHITE))
        black_knight = len(state.board.pieces(chess.KNIGHT, chess.BLACK))
        white_bishop = len(state.board.pieces(chess.BISHOP, chess.WHITE))
        black_bishop = len(state.board.pieces(chess.BISHOP, chess.BLACK))
        white_queen = len(state.board.pieces(chess.QUEEN, chess.WHITE))
        black_queen = len(state.board.pieces(chess.QUEEN, chess.BLACK))
        white_king = len(state.board.pieces(chess.KING, chess.WHITE))
        black_king = len(state.board.pieces(chess.KING, chess.BLACK))
        white_rook = len(state.board.pieces(chess.ROOK, chess.WHITE))
        black_rook = len(state.board.pieces(chess.ROOK, chess.BLACK))
        white_pawn = len(state.board.pieces(chess.PAWN, chess.WHITE))
        black_pawn = len(state.board.pieces(chess.PAWN, chess.BLACK))

        material = self.knightweight * (white_knight - black_knight) + self.bishopweight * (white_bishop - black_bishop) + self.queenweight * (white_queen - black_queen) + self.kingweight * (white_king - black_king) + self.rookweight * (white_rook - black_rook) + self.pawnweight * (white_pawn - black_pawn)
        
        pinned_val = evaluate_pinned(state.board.pieces(chess.KNIGHT, chess.WHITE), chess.WHITE, self.knight_pinned_value) + evaluate_pinned(state.board.pieces(chess.KNIGHT, chess.WHITE), chess.BLACK, -self.knight_pinned_value) +\
                        evaluate_pinned(state.board.pieces(chess.BISHOP, chess.WHITE),chess.WHITE, self.bishop_pinned_value) + evaluate_pinned(state.board.pieces(chess.BISHOP, chess.BLACK),chess.BLACK, -self.bishop_pinned_value) +\
                        evaluate_pinned(state.board.pieces(chess.QUEEN, chess.WHITE),chess.WHITE, self.queen_pinned_value) + evaluate_pinned(state.board.pieces(chess.QUEEN, chess.BLACK),chess.BLACK, -self.queen_pinned_value)                 

        attacking_val = attacking_value(state.board.pieces(chess.KNIGHT, chess.WHITE), [state.board.pieces(chess.KNIGHT, chess.BLACK), state.board.pieces(chess.BISHOP, chess.BLACK), state.board.pieces(chess.QUEEN, chess.BLACK)], self.knight_attacking_value) +\
                        attacking_value(state.board.pieces(chess.KNIGHT, chess.BLACK), [state.board.pieces(chess.KNIGHT, chess.WHITE), state.board.pieces(chess.BISHOP, chess.WHITE), state.board.pieces(chess.QUEEN, chess.WHITE)], self.black_knight_attacking_value) +\
                        attacking_value(state.board.pieces(chess.BISHOP, chess.WHITE), [state.board.pieces(chess.KNIGHT, chess.BLACK), state.board.pieces(chess.BISHOP, chess.BLACK), state.board.pieces(chess.QUEEN, chess.BLACK)], self.bishop_attacking_value) +\
                        attacking_value(state.board.pieces(chess.BISHOP, chess.BLACK), [state.board.pieces(chess.KNIGHT, chess.WHITE), state.board.pieces(chess.BISHOP, chess.WHITE), state.board.pieces(chess.QUEEN, chess.WHITE)], self.black_bishop_attacking_value) +\
                        attacking_value(state.board.pieces(chess.QUEEN, chess.WHITE), [state.board.pieces(chess.KNIGHT, chess.BLACK), state.board.pieces(chess.BISHOP, chess.BLACK), state.board.pieces(chess.QUEEN, chess.BLACK)], self.queen_attacking_value) +\
                        attacking_value(state.board.pieces(chess.QUEEN, chess.BLACK), [state.board.pieces(chess.KNIGHT, chess.WHITE), state.board.pieces(chess.BISHOP, chess.WHITE), state.board.pieces(chess.QUEEN, chess.WHITE)], self.black_queen_attacking_value)
                        
        white_mobility = mobility_evaluation(state, chess.WHITE)
        black_mobility = mobility_evaluation(state, chess.BLACK)
        mobility_score = (white_mobility - black_mobility)
        
        eval = material + pinned_val * 0.1 + attacking_val * 0.1 + mobility_score * self.mobility_multiplier
        
        #This chess variation has strange rules that need to be taken into account
        
        
        if state.board.is_checkmate():
            eval += 9998
        
        if not is_white:
            eval = -eval

        return eval

    def decide(self, state: AbstractState):
        
        if state.current_player() == 0 and state.board.fullmove_number == 1:
            # First move as white
            # But sadly we don't know the tournament board setup, so we can't hardcode the first move to start funny lines :(
            self.side = "white"
            """
            chessmove = chess.Move.from_uci("b1c2")
            #chessmove = chess.Move.from_uci("a1b3")
            #chessmove = chess.Move.from_uci("d1b3")
            action = Action(chessmove)
            yield action
            return
            """
        if state.current_player() == 1 and state.board.fullmove_number == 1:
            self.side = "black"
        depth = 4
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
