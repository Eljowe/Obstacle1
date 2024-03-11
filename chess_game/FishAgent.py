from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np
import json


latest = {
    "score": [1, 0],
    "all_scores": [11, 1],
    "bishopstable": [
      33.97220230102539, 14.701160430908203, 111.21765899658203, 40.29605484008789, -4.5214385986328125,
      22.77438735961914, -130.47310829162598, -11.123706817626953, 20.218597412109375, 77.90174102783203,
      39.47212219238281, -51.119449615478516, 44.25175094604492, -4.137859344482422, -52.57790184020996,
      -40.251007080078125, 31.28008270263672, -158.10285186767578, -121.53617095947266, -9.913406372070312,
      -101.45183372497559, 18.671628952026367, -102.34074783325195, -32.868350982666016, 24.126602172851562
    ],
    "knightstable": [
      -47.401615142822266, 112.36600494384766, -18.64327621459961, 14.23710823059082, -12.985431671142578,
      -120.5858154296875, 150.77571868896484, 33.593570709228516, -35.46089553833008, 71.35166549682617,
      -32.11539268493652, -0.8450546264648438, 99.08062362670898, 224.89987564086914, -25.90169334411621,
      -48.723249435424805, -35.77323532104492, -106.00259590148926, 15.665773391723633, 32.36197471618652,
      -28.301692962646484, -117.77738761901855, 65.68376922607422, -21.29129409790039, -197.95860290527344
    ],
    "queenstable": [
      -40.744606018066406, -7.488500595092773, 44.780216217041016, 36.891456604003906, 39.99561309814453,
      -38.12228775024414, 8.303831100463867, -11.54275131225586, 102.31912231445312, 5.805301666259766,
      2.1414260864257812, 12.110401153564453, -24.27361297607422, -6.170541763305664, 15.233785629272461,
      -87.57240676879883, -61.36701583862305, -13.183929443359375, 59.401113510131836, 60.54607582092285,
      14.3724365234375, -24.251955032348633, 49.452388763427734, 62.17961502075195, -97.21824645996094
    ],
    "kingstable": [
      26.669084548950195, 114.7099838256836, -97.15562438964844, -1.1336479187011719, -30.46630859375,
      -127.20301818847656, -31.700593948364258, -15.628181457519531, 38.348859786987305, -129.39426803588867,
      11.55023193359375, 38.0172233581543, 49.15434265136719, -138.12114143371582, 158.32180404663086,
      -47.201175689697266, -94.94905090332031, 138.5243148803711, -12.647357940673828, -126.58427429199219,
      70.71862411499023, -78.87612533569336, -7.241241455078125, 32.15292167663574, -74.74973678588867
    ],
    "bishopweight": 404.77230644226074,
    "knightweight": 350.68303298950195,
    "queenweight": 743.7789154052734,
    "kingweight": -28.104129791259766
  }

class FishAgent():
    """
    The interface of an Agent

    This class defines the required methods for an agent class
    """
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.__player = None
        self.side = None
        
        with open('tables.json', 'r') as f:
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
            "agent name": "FishyAgent",
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
