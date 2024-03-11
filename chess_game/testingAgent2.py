from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID
import numpy as np




class TestingAgent2():
    """
    The interface of an Agent

    This class defines the required methods for an agent class
    """
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.__player = None
        self.side = None
        self.knightweight = -362.5022945404053
        self.bishopweight = 797.1532001495361
        self.queenweight = 2296.3114490509033
        self.kingweight = -736.4896087646484
    #Change these values with deep learning
    #Also find the best first moves with deep learning
    #Or even create an opening theory with DL
        self.bishopstable = self.reverse_table_reshape([
      -501.1955814361572, -182.75673866271973, -46.36678886413574, -1119.1163654327393, -917.6037940979004,
      -305.0183525085449, 1192.0362758636475, -784.9645519256592, -1219.201244354248, -926.8753852844238,
      1020.2403583526611, -607.5099277496338, -557.3279228210449, -812.3737239837646, -746.0209083557129,
      565.7762088775635, -1002.9312324523926, 210.65349769592285, -454.57827949523926, 271.8985290527344,
      401.98733139038086, 301.5482234954834, 53.88715362548828, -1019.2499485015869, -205.23379516601562
    ])
        
        
        self.knightstable = self.reverse_table_reshape([
      -646.1920394897461, 220.23149490356445, 971.5422554016113, -1007.2208957672119, 542.3589363098145,
      -778.8246574401855, -304.00771713256836, -379.7312259674072, 37.075674057006836, 155.53550338745117,
      368.12135124206543, -430.2525119781494, -420.5830192565918, -107.31211471557617, 313.6958541870117,
      -1090.492727279663, -1411.3888835906982, -42.641273498535156, 1112.550256729126, 86.22482299804688,
      296.79910469055176, 48.24403190612793, 284.29502296447754, 584.1754341125488, 470.508508682251
    ])
        
        
        self.queenstable = self.reverse_table_reshape([
      89.02659606933594, 626.0453357696533, 125.44550704956055, -1443.2975692749023, -230.98684310913086,
      -223.49601936340332, 829.4441375732422, 993.7721862792969, 0.4023017883300781, -853.8765354156494,
      -1287.878246307373, -43.264198303222656, 138.3327693939209, 841.9679679870605, 185.46385955810547,
      -337.49399185180664, 183.37460708618164, -518.6640777587891, -499.52788734436035, -160.9960594177246,
      168.9617862701416, -329.9191150665283, -966.2401714324951, -173.09992790222168, 1019.3602294921875
    ])
        
        self.kingstable = self.reverse_table_reshape([
      1670.8158950805664, -286.38355827331543, 554.476972579956, 343.69029426574707, -1761.2886295318604,
      -773.4627075195312, 526.289945602417, -165.27223205566406, 580.133581161499, -1569.8294620513916,
      -1055.719404220581, 1287.9407444000244, 178.31885719299316, -40.10418510437012, 5.687213897705078,
      202.80259132385254, 938.0516700744629, -389.2188129425049, 694.8278675079346, 342.2439498901367,
      -75.42897033691406, 512.9526138305664, -471.559268951416, -1583.1664810180664, -70.73262786865234
    ])

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
            "agent name": "Testing agent2",
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
