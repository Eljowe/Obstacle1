from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np


"""
This agent is a chess agent using the minimax algorithm with alpha-beta pruning and quiesce search.
The move-deciding algorithm is a simple implementation, and has no further optimizations or depth variations.

The algorithm uses a custom evaluation function to evaluate the board state.
The evaluation function is based on the material difference and the piece-square tables.

The agent also uses weights for each piece and piece-square table to evaluate the board state.
The weights and tables are based on moderate amount of reinforcement learning training with stable baselines3,
where the goal was to optimize the evaluation function to win games against other agents, itself as well.

This wasn't the focus of this course, but I was interested in trying to combine the methods of the course with
reinforcement learning to see if I could improve the agent's performance after implementing the minimax algorithm.
"""

weights = {
    "bishopstable": [
      71.99599075317383, -77.51302146911621, -166.91599655151367, 32.34518241882324, 8.907079696655273,
      64.80254364013672, 45.897823333740234, 14.297367095947266, 105.33560562133789, 54.5963134765625, 3.5174560546875,
      71.97987747192383, -53.905935287475586, 8.67525863647461, 50.439409255981445, -26.886837005615234,
      99.76242256164551, 117.02140426635742, 43.500732421875, -53.911495208740234, 89.74701309204102, 32.3294563293457,
      55.35352325439453, 34.12389373779297, 11.144378662109375
    ],
    "knightstable": [
      69.57744789123535, 69.86259078979492, 1.3050365447998047, -114.62079048156738, 16.812782287597656,
      64.54365921020508, -56.933433532714844, -4.729099273681641, -24.892498016357422, 48.90412139892578,
      -74.06539535522461, 11.747297286987305, 29.60332489013672, 8.771347045898438, -23.9217472076416,
      92.47920989990234, -62.003658294677734, 12.810556411743164, -23.161657333374023, -75.61394119262695,
      70.90402221679688, 26.742033004760742, -2.1107349395751953, -39.32065963745117, -19.759761810302734
    ],
    "queenstable": [
      54.142967224121094, 24.220565795898438, -15.047012329101562, -56.22271537780762, -66.84842872619629,
      11.62957763671875, -52.240203857421875, 11.551473617553711, 3.4086036682128906, 61.6417121887207,
      -35.56008338928223, 64.71196556091309, 56.33319091796875, 2.273405075073242, -51.341468811035156,
      13.735006332397461, 73.46980285644531, 2.7376022338867188, 88.84739303588867, -34.89010429382324,
      -200.67144393920898, -26.639530181884766, 94.29778289794922, 23.317113876342773, 51.66970634460449
    ],
    "kingstable": [
      105.51416778564453, 7.555662155151367, 4.674459457397461, 224.3195343017578, -4.602453231811523,
      -67.89537048339844, 23.074504852294922, -96.86635971069336, -112.1214599609375, -84.50157737731934,
      -18.09286880493164, -37.681636810302734, -82.82785606384277, 105.30342864990234, -27.467479705810547,
      -76.98773956298828, -38.18513870239258, -58.98675346374512, -32.64807319641113, -79.19097328186035,
      -59.974056243896484, -90.1920337677002, 79.98091888427734, 10.42558479309082, 33.541215896606445
    ],
    "bishopweight": 336.3208827972412,
    "knightweight": 294.1728858947754,
    "queenweight": 1024.7459907531738,
    "kingweight": 65.03753089904785
  }

class Agent():
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.__player = None
        self.side = None
        self.knightweight = weights["knightweight"]
        self.bishopweight = weights["bishopweight"]
        self.queenweight = weights["queenweight"]
        self.kingweight = weights["kingweight"]
        
        self.bishopstable = self.reverse_table_reshape(weights["bishopstable"])
        self.knightstable = self.reverse_table_reshape(weights["knightstable"])
        self.queenstable = self.reverse_table_reshape(weights["queenstable"])
        self.kingstable = self.reverse_table_reshape(weights["kingstable"])

    def reverse_table_reshape(self, table):
        #Needed for resizing 5x5 table to 8x8
        board_2d = [table[i:i+5] for i in range(0, len(table), 5)]
        original_board_2d = [row + [0]*3 for row in board_2d] + [[0]*8]*3
        original_board = [cell for row in original_board_2d for cell in row]
        return original_board

    @staticmethod
    def info():
        return {
            "agent name": "Obstacle1",
            "description": "A chess agent using the minimax algorithm with alpha-beta pruning and quiesce search",
        }

    
    def alphabeta(self, alpha, beta, depthleft, state: State):
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
        
        eval = material + knight_eval + bishop_eval + queens_eval + kings_eval
        if id == 0:
            return eval
        else:
            return -eval


    def decide(self, state: AbstractState):
        bestValue = -99999
        alpha = -100000
        beta = 100000
        moves = state.applicable_moves()
        random.shuffle(moves)
        best_action = moves[0]
        for action in moves:
            state.execute_move(action)
            action_value = -self.alphabeta(-beta, -alpha, self.depth - 1, state)
            
            if action_value > bestValue:
                bestValue = action_value
                best_action = action
                yield best_action
            if action_value > alpha:
                alpha = action_value
            state.undo_last_move()
        yield best_action

    def __str__(self):
        return self.info()["agent name"]
