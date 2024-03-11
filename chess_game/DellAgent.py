from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np
import json

working = {
    "score": [1, 0],
    "all_scores": [11, 1],
    "bishopstable": [
      33.99340057373047, 139.20833778381348, -122.50811386108398, 180.77107048034668, -84.27004623413086,
      71.4963150024414, 22.430452346801758, 25.04399871826172, -60.88835525512695, 147.45465469360352,
      60.90236282348633, 38.04176712036133, 108.75913619995117, 137.3330020904541, -101.40392112731934,
      -32.99343490600586, -34.071895599365234, -108.18863868713379, -118.24479866027832, 131.82544326782227,
      -54.44297790527344, -94.76387977600098, -49.8341178894043, 18.419401168823242, 13.599971771240234
    ],
    "knightstable": [
      -43.610252380371094, -91.01543045043945, 13.286367416381836, -5.9279327392578125, -180.06548690795898,
      -66.2959976196289, -69.11059761047363, 20.871305465698242, 5.297412872314453, 26.015249252319336,
      217.99596786499023, 160.7254524230957, -50.60584831237793, 54.86085319519043, -6.95880126953125,
      -142.89286041259766, 22.411495208740234, 152.44526290893555, -60.90302658081055, -132.3129768371582,
      13.352628707885742, -8.62580680847168, 108.68336868286133, -43.02564239501953, 9.833209991455078
    ],
    "queenstable": [
      46.68845748901367, 114.98604202270508, 145.22750854492188, -175.30591583251953, -17.03770637512207,
      -98.57176208496094, -1.3035659790039062, -128.8183135986328, 50.500144958496094, -62.206687927246094,
      -30.41640281677246, -66.05207824707031, 12.946224212646484, 63.56924629211426, 57.062211990356445,
      -146.6356029510498, -18.982954025268555, -49.499542236328125, 46.98201942443848, 113.74290466308594,
      -29.191211700439453, -28.853914260864258, -0.8882427215576172, 290.9601936340332, 145.43188667297363
    ],
    "kingstable": [
      85.15066909790039, -146.31482696533203, 67.79450416564941, 62.842994689941406, 105.04769325256348,
      -108.93074417114258, -124.15279388427734, -114.32601928710938, -18.05683135986328, -116.98903465270996,
      10.559032440185547, -114.39245986938477, 27.055355072021484, -149.98235321044922, 188.10487747192383,
      -43.2814998626709, -121.91914367675781, 82.51213073730469, -75.14758110046387, 17.77452850341797,
      -41.619728088378906, -193.27106285095215, -9.431550979614258, -35.458534240722656, -153.9565486907959
    ],
    "bishopweight": 471.2348442077637,
    "knightweight": 489.2557945251465,
    "queenweight": 767.3705558776855,
    "kingweight": 212.3141040802002
  }


class DellAgent():
    """
    The interface of an Agent

    This class defines the required methods for an agent class
    """
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.__player = None
        self.side = None
        
        with open('delltables.json', 'r') as f:
            tables = json.load(f)
        
        self.knightweight = tables[-1]['knightweight']
        self.bishopweight = tables[-1]['bishopweight']
        self.queenweight = tables[-1]['queenweight']
        self.kingweight = tables[-1]['kingweight']
        
        
        self.bishopstable = self.reverse_table_reshape(tables[-1]['bishopstable'])
        self.knightstable = self.reverse_table_reshape(tables[-1]['knightstable'])
        self.queenstable = self.reverse_table_reshape(tables[-1]['queenstable'])
        self.kingstable = self.reverse_table_reshape(tables[-1]['kingstable'])
        
        

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
        """
        Return the agent's information

        This function returns the agent's information as a dictionary variable.
        The returned dictionary should contain at least the `agent name`.

        Returns
        -------
        Dict[str, str]
        """
        return {
            "agent name": "DellAgent",
        }
        raise NotImplementedError

    
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
        
        eval = material + knight_eval + bishop_eval + queens_eval + kings_eval
        if id == 0:
            return eval
        else:
            return -eval


    def decide(self, state: AbstractState):
        """
        Generate a sequence of increasing good actions

        NOTE: You can find the possible actions from `state` by calling
              `state.successors()`, which returns a list of pairs of
              `(action, successor_state)`.

        This is a generator function; it means it should have no `return`
        statement, but it should `yield` a sequence of increasing good
        actions.

        Parameters
        ----------
        state: State
            Current state of the game

        Yields
        ------
        action
            the chosen `action` from the `state.successors()` list
        """
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
        #print("best value: ", bestValue)
        #print("side: ", self.side)
        #print("best action: ", best_action)
        #print("custom evaluate: ", self.custom_evaluate_board(state))
        yield best_action

    def __str__(self):
        return self.info()["agent name"]
