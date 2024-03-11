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
        
        self.bishopweight = 326
        self.knightweight = 385
        self.queenweight = 950
        self.kingweight = 5000 
    #Change these values with deep learning
    #Also find the best first moves with deep learning
    #Or even create an opening theory with DL
        self.bishopstable = self.reverse_table_reshape([
      -8.257211685180664, -45.602325439453125, -66.8542251586914, -70.00279998779297, 30.53160858154297,
      -22.614757537841797, -1.6128463745117188, -53.65303039550781, 12.89461898803711, -43.34302520751953,
      79.06857299804688, 48.47069549560547, 66.27078819274902, 4.720771789550781, -91.89281845092773,
      -5.513801574707031, -1.6561660766601562, 3.6916847229003906, -99.49597549438477, -8.166702270507812,
      54.079071044921875, -28.698482513427734, -33.59449768066406, 17.429420471191406, -29.410907745361328
    ])
        
        
        self.knightstable = self.reverse_table_reshape([
      -10.288726806640625, -125.86349868774414, 62.52924728393555, -76.05221748352051, -3.473358154296875,
      100.36528778076172, -34.11406326293945, 101.96412658691406, -45.1275577545166, 18.027130126953125,
      81.38987731933594, -80.70923233032227, 15.510929107666016, 45.97636032104492, 85.35476684570312,
      53.09674072265625, -41.41509246826172, 42.96012496948242, -30.750375747680664, -18.808902740478516,
      -57.731224060058594, -45.02654838562012, 34.589115142822266, 40.57453155517578, -53.474822998046875
    ])
        
        
        self.queenstable = self.reverse_table_reshape([
      -74.46636962890625, -36.53805160522461, 55.39234924316406, -64.91791343688965, 11.802711486816406,
      71.38187408447266, 7.890228271484375, 47.268089294433594, -24.857173919677734, -70.12844848632812,
      39.94581985473633, -0.6905097961425781, -34.599958419799805, -36.67597198486328, 23.263694763183594,
      -17.507102966308594, 18.670448303222656, -31.96295928955078, 43.65176010131836, -9.001533508300781,
      -37.99224281311035, -58.39117431640625, 7.3231964111328125, 36.93107986450195, -78.52197265625
    ])
        
        self.kingstable = self.reverse_table_reshape([
      71.41886520385742, 34.094520568847656, -10.11540412902832, 26.197128295898438, -9.944339752197266,
      -92.38948440551758, 7.463493347167969, -46.049997329711914, -55.666175842285156, -136.0878028869629,
      16.32215118408203, -26.00394058227539, 10.413810729980469, -0.5139312744140625, -3.4421463012695312,
      -48.62348556518555, -29.396961212158203, -75.65679550170898, 44.776466369628906, 59.63180160522461,
      46.36799621582031, -17.97539520263672, 22.904151916503906, -108.77283477783203, -51.906105041503906
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
        
        def evaluate_pinned(piece_set, color, value_of_pin):
            eval = 0
            for piece in piece_set:
                if state.board.is_pinned(color, piece):
                    eval = eval + value_of_pin
            return eval
        
        pinned_value = evaluate_pinned(state.board.pieces(chess.KNIGHT, chess.WHITE), chess.WHITE, -30) + evaluate_pinned(state.board.pieces(chess.KNIGHT, chess.WHITE), chess.BLACK, 30) +\
                        evaluate_pinned(state.board.pieces(chess.BISHOP, chess.WHITE),chess.WHITE, -30) + evaluate_pinned(state.board.pieces(chess.BISHOP, chess.BLACK),chess.BLACK, 30) +\
                        evaluate_pinned(state.board.pieces(chess.QUEEN, chess.WHITE),chess.WHITE, -90) + evaluate_pinned(state.board.pieces(chess.QUEEN, chess.BLACK),chess.BLACK, 90)
                            
        
        
        eval = material + knight_eval + bishop_eval + queens_eval + kings_eval + pinned_value
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
