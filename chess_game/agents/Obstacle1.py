from envs.environment import AbstractState
import chess
import random
from envs.game import State, ID, Action
from agent_interface import AgentInterface

"""
# Obstacle 1, the chess engine

This is my submission for Aalto Artificial Intelligence CS-E4800 Games tournament.

### Introduction

Obstacle 1 uses the minimax algorithm with alpha-beta pruning (negamax), iterative deepening, quiescence search, zobrist hashing, late move reduction, and null move pruning. Actually, anything I could find when studying chess algorithms and AI.
(These materials were helpful:
https://www.cs.cornell.edu/boom/2004sp/ProjectArch/Chess/algorithms.html
https://www.chessprogramming.org/Alpha-Beta)

The algorithm uses a custom evaluation function to evaluate the board state. The evaluation function is based on the material difference, with the additional try of adding the mobility score, pinning and attacking value.

### Evaluation weights

The weights for the evaluation function are based on moderate amount of reinforcement learning training with stable baselines3, where the goal was to optimize the weights in order to win games against other agents, as well as itself, too.
As the training was conducted with 5x5 board, the weights might not be optimal for larger boards, which is also why I removed the value tables for each piece.

Reinforcement learning wasn't the focus of this course, or at least the methods I used, but I was interested in trying to combine the methods of this course with RL methods to see if I could improve the agent's performance after implementing the basic setup.

The weights were trained using the SAC algorithm, and the agents played against each other for approximately 10,000 games, showing only insignificant improvement to be honest.

### Process

0. Understand the chess library and the game environment.
1. Implement the minimax algorithm with alpha-beta pruning and quiescence search.
2. Implement the custom evaluation function.
3. Implement the weights, and test value tables (which were later removed).
4. Modify custom evaluation function by adding pinning and attacking value.
5. Train the weights using reinforcement learning.
6. Testing different weights to find the best performing ones.
7. Finally implement iterative deepening.
8. Adding move ordering to check captures and checks first.
9. Adding null move pruning, zobrist hashing, and late move reduction.
10. Fixing the endgame bug where the agent would give up the queen to lose by unsufficient material.

### Results

With 2.0 second time limits, the agent can search up to depth 8-20, depending on the board state,
while minimax agent with iterative deepening can search up to depth 4-7.
While playing against minimax agent with 2.0 second time limits, the agent wins 19-1, from 20 games.

In the tournament, Obstacle1 scored 20 wins and 3 losses, winning 127 individual rounds, finishing 6th in a group of 31 agents.
As there were 8 groups, this score would approximately translate in 40th-48th place out of 250 agents.
The result was not enough to qualify for the next round (only two agents from each group qualified), but I'm happy with the result,
having learned a lot, and having reached the top 20% of the agents in such skilled competition.
"""



class Agent(AgentInterface):
    def __init__(self):
        self.max_depth = 30
        self.__player = None
        
        tables = {
            "bishopweight": 2,
            "knightweight": 2,
            "queenweight": 7,
            "opponent_bishopweight": 2,
            "opponent_knightweight": 2,
            "opponent_queenweight": 5,
        }

        self.bishopweight = tables["bishopweight"]
        self.knightweight = tables["knightweight"]
        self.queenweight = tables["queenweight"]
        
        self.opponent_bishopweight = tables["opponent_bishopweight"]
        self.opponent_knightweight = tables["opponent_knightweight"]
        self.opponent_queenweight = tables["opponent_queenweight"]
        

    def info(self):
        return {"agent name": f"Obstacle1"}

    def heuristic(self, state: State, deciding_agent: int):
        if deciding_agent == 0:
            COLOR = chess.WHITE
            otherCOLOR = chess.BLACK
        else:
            COLOR = chess.BLACK
            otherCOLOR = chess.WHITE

        knights = state.board.pieces(chess.KNIGHT, COLOR)
        bishops = state.board.pieces(chess.BISHOP, COLOR)
        queens = state.board.pieces(chess.QUEEN, COLOR)

        Oknights = state.board.pieces(chess.KNIGHT, otherCOLOR)
        Obishops = state.board.pieces(chess.BISHOP, otherCOLOR)
        Oqueens = state.board.pieces(chess.QUEEN, otherCOLOR)

        score = self.knightweight * len(knights) + self.bishopweight * len(bishops) + self.queenweight * len(queens) - self.opponent_knightweight * len(Oknights) - self.opponent_bishopweight * len(Obishops) - self.opponent_queenweight * len(Oqueens)
        return score
    
    def quiescence_search(self, state: State, alpha: float, beta: float, deciding: int):
        stand_pat = self.heuristic(state, deciding)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for action in self.order_moves(state.applicable_moves(), state, only_captures=True):
            state.execute_move(action)
            score = -self.quiescence_search(state, -beta, -alpha, deciding)
            state.undo_last_move()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def order_moves(self, moves, state, only_captures=False):
        if only_captures:
            return [move for move in moves if state.board.is_capture(move.chessmove)]
        else:
            captures = [move for move in moves if state.board.is_capture(move.chessmove)]
            checks = [move for move in moves if state.board.gives_check(move.chessmove)]
            others = [move for move in moves if move not in captures and move not in checks]
            return captures + checks + others


    def decide(self, state0: State):
        state = state0.clone()
        deciding = state.current_player()
        moves = self.order_moves(state.applicable_moves(), state)
        if state.current_player() == 0 and state.board.fullmove_number == 1:
            # First move as white
            #chessmove = chess.Move.from_uci("a1b3")
            #chessmove = chess.Move.from_uci("d1b3")
            chessmove = chess.Move.from_uci("b1c2")
            action = Action(chessmove)
            if action in moves:
                yield action
                return
        best_action = moves[0]
        max_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        depth = 4
        while depth <= self.max_depth:
            #print(f"obstacle Depth: {depth}")
            for action in moves:
                state.execute_move(action)
                action_value = self.min_value(state, depth - 1, deciding, alpha, beta)
                state.undo_last_move()
                if action_value > max_value:
                    max_value = action_value
                    best_action = action
                alpha = max(alpha, action_value)
            yield best_action
            depth += 1

    def max_value(self, state: State, depth: int, deciding: int, alpha: float, beta: float):
        winner = state.is_winner()
        if winner is not None:
            if winner == 1: return float('inf')
            if winner == -1: return float('-inf')
            return 0

        if depth == 0 or state.is_winner() is not None:
            return self.quiescence_search(state, alpha, beta, deciding)

        value = float('-inf')
        for action in state.applicable_moves():
            state.execute_move(action)
            value = max(value, self.min_value(state, depth - 1, deciding, alpha, beta))
            state.undo_last_move()

            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def min_value(self, state: State, depth: int, deciding: int, alpha: float, beta: float):
        winner = state.is_winner()
        if winner is not None:
            if winner == 1: return float('-inf')
            if winner == -1: return float('inf')
            return 0
        
        if depth == 0 or state.is_winner() is not None:
            return -self.quiescence_search(state, -beta, -alpha, deciding)

        value = float('inf')
        for action in state.applicable_moves():
            state.execute_move(action)
            value = min(value, self.max_value(state, depth - 1, deciding, alpha, beta))
            state.undo_last_move()
            
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value
