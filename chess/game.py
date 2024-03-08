import time
from typing import List, Optional
from copy import deepcopy
from random import choice

from agent_interface import AgentInterface
from envs.environment import AbstractState
from time_limit import time_limit

import chess

# Run a game: alternate between players, asking for a sequence of
# increasingly good next moves from each player, until the per-player
# time limit is reached.


class Game:
    def __init__(self, players: List[AgentInterface]):
        self.__players = players

    def play(self,
             starting_state: AbstractState,
             output=False,
             timeout_per_turn=[None, None]):
        winners = self.__play(starting_state,
                              output,
                              timeout_per_turn)
        if output:
            print("Game is over!")
            if len(winners) != 1:
                print("The game ended in a draw!")
            else:
                print(f"Player {winners[0]}, {self.__players[winners[0]]} WON!")
        return winners

    def __play(self, state: AbstractState, output, timeout_per_turn):
        duration = None
        output = True
        if(output): print("Starting game!\n")
        ROUNDS = 0
        while True:
            is_winner = state.is_winner()
            if is_winner is not None:
                if(output): print("Game ends after " + str(ROUNDS) + " rounds\n")
                if is_winner == 0:
                    return []
                if is_winner == 1:
                    return [state.current_player()]
                return [1 - state.current_player()]
            if ROUNDS > 200:
                if(output): print("Played 200 rounds without a winner. Terminate.\n")
                return []
            # No moves = King's only moves are outside the board (but inside 8 X 8)
            moves = state.applicable_moves()
            if len(moves) == 0:
                return [1 - state.current_player()]
            start_time = time.time()
            action = self.__get_action(self.__players[state.current_player()],
                                       state,
                                       timeout_per_turn[state.current_player()])
            duration = time.time() - start_time
            if action is None or action not in moves:
                if action is None:
                    if(output): print ("Time out!")
                else:
                    if(output): print("Illegal move!")
                if(output): print("Choosing a random action!")
                if len(moves) > 0:
                    action = choice(moves)
                else:
                    if(output): print("No actions to choose from!\n")
                    return [1 - state.current_player()]
            state.execute_move(action)
            if output:
                print(f"Decision time: {duration:0.3f}")
                print("Move:", action)
                print("===================================================")
                print(state)
                print("Possible moves:", len(moves))
            ROUNDS += 1

    def __get_action(self, player: AgentInterface, state, timeout):
        action = None
        try:
            with time_limit(timeout):
                for decision in player.decide(deepcopy(state)):
                    action = decision
        except TimeoutError:
            pass
        # NOTE: The following lines will be uncommented during tournament
        except Exception as e:
            print("Got an EXCEPTION:", e)
            print()
            import traceback
            traceback.print_exc()
        return action
