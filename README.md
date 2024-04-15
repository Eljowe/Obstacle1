# Obstacle 1, the chess engine

This is my submission for the Aalto Artificial Intelligence CS-E4800 Games tournament.

### Introduction

Obstacle 1 uses the minimax algorithm with alpha-beta pruning (negamax), iterative deepening, quiescence search, Zobrist hashing, late move reduction, and null move pruning. Actually, anything I could find when studying chess algorithms and AI.
(These materials were helpful:
https://www.cs.cornell.edu/boom/2004sp/ProjectArch/Chess/algorithms.html
https://www.chessprogramming.org/Alpha-Beta)

The algorithm uses a custom evaluation function to evaluate the board state. The evaluation function is based on the material difference, with the additional try of adding the mobility score, pinning, and attacking value.

### Evaluation weights

The weights for the evaluation function are based on a moderate amount of reinforcement learning training with stable baselines3, where the goal was to optimize the weights in order to win games against other agents, as well as itself, too.
As the training was conducted with a 5x5 board, the weights might not be optimal for larger boards, which is also why I removed the value tables for each piece.

Reinforcement learning wasn't the focus of this course, or at least the methods I used, but I was interested in trying to combine the methods of this course with RL methods to see if I could improve the agent's performance after implementing the basic setup.

The weights were trained using the SAC algorithm, and the agents played against each other for approximately 10,000 games, showing only insignificant improvement to be honest.

### Process

0. Understand the chess library and the game environment.
1. Implement the minimax algorithm with alpha-beta pruning and quiescence search.
2. Implement the custom evaluation function.
3. Implement the weights, and test value tables (which were later removed).
4. Modify the custom evaluation function by adding pinning and attacking values.
5. Train the weights using reinforcement learning.
6. Testing different weights to find the best-performing ones.
7. Finally implement iterative deepening.
8. Adding move ordering to check captures and checks first.
9. Adding null move pruning, Zobrist hashing, and late move reduction.
10. Fixing the endgame bug where the agent would give up a queen to lose by insufficient material.

### Results

With 2.0 second time limits, the agent can search up to depth 8-20, depending on the board state,
while the minimax agent with iterative deepening can search up to depths 4-7.
While playing against a minimax agent with 2.0 second time limits, the agent wins 19-1, from 20 games.
The strength will be further determined in the upcoming tournament.

### Tournament results

In the tournament, Obstacle1 scored 20 wins and 3 losses, winning 127 individual rounds, finishing 6th in a group of 31 agents.
As there were 8 groups, this score would approximately translate in 40th-48th place out of 250 agents.
The result was not enough to qualify for the next round (only two agents from each group qualified), but I'm happy with the result,
having learned a lot, and having reached the top 20% of the agents in such skilled competition.
