from random import shuffle
from agent_interface import AgentInterface
from envs.game import State
from random_agent import RandomAgent
from game import Game


class MCSAgent(AgentInterface):
    """
    Evaluate each action by taking it, followed by
    random plays. The action with most wins is chosen.
    """
    def __init__(self):
        self.__simulator = Game([RandomAgent(), RandomAgent()])

    def info(self):
        return {"agent name": "MCS"}

    def decide(self, state: State):
        moves = state.applicable_moves()
        shuffle(moves)
        win_counter = [0] * len(moves)
        while True:
            for i, m in enumerate(moves):
                next_state = state.clone()
                next_state.execute_move(m)
                result = self.__simulator.play(output=False, 
                                               starting_state=next_state)
                win_counter[i] += 1 if result == [state.current_player()] else 0
            yield moves[win_counter.index(max(win_counter))]
