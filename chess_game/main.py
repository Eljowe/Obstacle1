from typing import Type
from random import seed

from game import Game
from envs.game import State

# Importing Agents
from agent_interface import AgentInterface
from agents.custom_agent import CustomAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.Obstacle2 import Agent2
from agents.Obstacle1 import Agent
# from agent import Agent    # After completing your agent, you can uncomment this line


# If you want the reproducibility uncomment the following line
# seed(13731367)


def main():
    ############### Set the players ###############
    #players = [Agent, MinimaxAgent]
    players = [Agent, MinimaxAgent]
    #players = [Agent, DLAgent]
    ###############################################

    RENDER = True

    # The rest of the file is not important; you can skip reading it. #
    ###################################################################
    results = [0, 0]
    for i in range(10):
        initial_state = State([player_name(p) for p in players])

        for round in range(len(players)):
            print( "########################################################")
            print("#{: ^54}#".format(f"ROUND {round}"))
            print( "########################################################")
            print( player_name(players[0]) + " is playing WHITE.")
            print( player_name(players[1]) + " is playing BLACK.")
            players_instances = [p() for p in players]
            # Timeout for each move. Don't rely on the value of it. This
            # value might be changed during the tournament.
            timeouts = [5, 5]
            game = Game(players_instances)
            new_round = initial_state.clone()
            turn_duration_estimate = sum([t
                                          for p, t in zip(players, timeouts)
                                          if p != RandomAgent])
            if RENDER:
                print(str(new_round))

            winners = game.play(new_round,
                                output=True,
                                timeout_per_turn=timeouts)
            if len(winners) == 1:
                results[winners[0]] += 1

            print()
            print(f"{i}) Result) {player_name(players[0])}: {results[0]} - "
                f"{player_name(players[1])}: {results[1]}")
            print("########################################################")

            # Rotating players for the next rounds
#            initial_state.rotate_players()
            players.append(players.pop(0))
            results.append(results.pop(0))


def player_name(player: Type[AgentInterface]):
    return player().info()['agent name']


if __name__ == "__main__":
    import platform
    if platform.system() == "Darwin":
        import multiprocessing
        multiprocessing.set_start_method('spawn')

    try:
        main()
    except BrokenPipeError as e:
        print("Broken Pipe Error:", e)
    except EOFError as e:
        print("EOF Error:", e)
