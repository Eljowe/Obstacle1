import random
from envs.environment import AbstractState
from agent_interface import AgentInterface


class RandomAgent(AgentInterface):
    @staticmethod
    def info():
        return {"agent name": "Random"}

    def decide(self, state: AbstractState):
        actions = state.applicable_moves()
        if(len(actions) == 0): return None
        yield random.choice(actions)
