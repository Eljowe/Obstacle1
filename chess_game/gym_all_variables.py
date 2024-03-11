import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import os
import numpy as np
import json

from typing import Type
from random import seed

from game import Game
from envs.game import State
from agent_interface import AgentInterface
from custom_agent import CustomAgent
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
from mcs_agent import MCSAgent
from testingAgent import TestingAgent
from DLAgent import DLAgent
from testingAgent2 import TestingAgent2

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import SAC
from gym import spaces

MAX_STEPS = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        
        self.agent = DLAgent()
        
        self.games_played = 0
        
        self.score = [0, 0]
        self.all_scores = [0,0 ]
        
        self.bishopstable = self.table_reshape(self.agent.bishopstable)
        self.knightstable = self.table_reshape(self.agent.knightstable)
        self.queenstable = self.table_reshape(self.agent.queenstable)
        self.kingstable = self.table_reshape(self.agent.kingstable)
        
        self.bishopweight = self.agent.bishopweight
        self.knightweight = self.agent.knightweight
        self.queenweight = self.agent.queenweight
        self.kingweight = self.agent.kingweight
        
        self.num_tables = 4
        self.table_size = 25
        self.score = 0
        self.action_space = spaces.Box(low=-50, high=50, shape=(4 * 25 + 4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(8,), dtype=np.float32)
    def table_reshape(self, table):
        original_board_2d = [table[i:i+8] for i in range(0, len(table), 8)]

        modified_board_2d = [row[:5] for row in original_board_2d[:5]]

        # Flatten the modified 2D board back to a 1D list
        modified_board = [cell for row in modified_board_2d for cell in row]
        return modified_board
    
    def reverse_table_reshape(self, table):
        # Convert the 1D list to a 2D list
        board_2d = [table[i:i+5] for i in range(0, len(table), 5)]

        # Expand the 2D list to an 8x8 board
        original_board_2d = [row + [0]*3 for row in board_2d] + [[0]*8]*3

        # Flatten the 2D board back to a 1D list
        original_board = [cell for row in original_board_2d for cell in row]
        return original_board
    
    def step(self, action):
        num_cells_per_table = self.table_size
        for i in range(self.num_tables):
            table = None
            if i == 0:
                table = self.bishopstable
            elif i == 1:
                table = self.knightstable
            elif i == 2:
                table = self.queenstable
            elif i == 3:
                table = self.kingstable

            for j in range(num_cells_per_table):
                action_value = action[i * num_cells_per_table + j]
                table[j] += action_value

        # Apply the last 4 actions to the weights
        self.bishopweight += action[-4]
        self.knightweight += action[-3]
        self.queenweight += action[-2]
        self.kingweight += action[-1]

        reward = self.calculate_reward()
        done = self.calculate_done()
        observation = reward
        truncated = False
        info = {"score": self.score, "games_played": self.games_played}
        return observation, reward, done, truncated, info
    
    def calculate_reward(self):
        # Play the game
        result = self.play_game()
        self.games_played += 1

        # Calculate the reward as the number of rounds won by the agent
        if result == 1:
            reward = self.score[0]**2
        elif result == -1:
            reward = -self.score[1]**2
        else:
            reward = -0.5
        return reward
    
    def reset(self, seed=None, options=None):
        # Reset the environment
        reset_info = {"score": self.score}  # Add any reset information you need here
        self.games_played = 0
        self.score = [0, 0]
        self.all_scores = [0, 0]
        return 0, reset_info
    
    def close(self):
        return super().close()
    
    def calculate_done(self):
        print(f"Games played: {self.games_played}")
        if self.games_played >= 1:
            print(f"Score: {self.score}")
            print(f"All scores: {self.all_scores}")
            print("\n")
            
            if self.all_scores[0] >= 11:
                print("Saving the tables to tables.json")
                with open('tables.json', 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:  # If the file is empty, set data to an empty list
                        data = []

                    # Append new data
                    data.append({
                        'score': self.score,
                        'all_scores': self.all_scores,
                        'bishopstable': self.bishopstable,
                        'knightstable': self.knightstable,
                        'queenstable': self.queenstable,
                        'kingstable': self.kingstable,
                        'bishopweight': self.bishopweight,
                        'knightweight': self.knightweight,
                        'queenweight': self.queenweight,
                        'kingweight': self.kingweight
                    })

                    # Write everything back to the file
                    with open('tables.json', 'w') as f:
                        json.dump(data, f)
        
            
            return True
        return False
    
    def play_game(self):
        ############### Set the players ###############
        #opponent = CustomAgent()
        opponent = TestingAgent2()
        players = [self.agent, opponent]
        self.agent.bishopstable = self.reverse_table_reshape(self.bishopstable)
        self.agent.knightstable = self.reverse_table_reshape(self.knightstable)
        self.agent.queenstable = self.reverse_table_reshape(self.queenstable)
        self.agent.kingstable = self.reverse_table_reshape(self.kingstable)
        self.agent.bishopweight = self.bishopweight
        self.agent.knightweight = self.knightweight
        self.agent.queenweight = self.queenweight
        self.agent.kingweight = self.kingweight
        
        #players = [AgentInterface, RandomAgent]
        #players = [MinimaxAgent, MinimaxAgent]
        #players = [MinimaxAgent, MCSAgent]
        #players = [RandomAgent, MCSAgent]
        #players = [RandomAgent, MinimaxAgent]
        #players = [RandomAgent, RandomAgent]
        #players = [MCSAgent, RandomAgent]

        # players = [Agent, IDMinimaxAgent]   <-- Uncomment this to test your agent
        ###############################################
        #table = np.array(self.table_reshape(self.agent.bishopstable))
        
        RENDER = False

        # The rest of the file is not important; you can skip reading it. #
        ###################################################################

        results = [0, 0]
        for i in range(6):
            initial_state = State([self.player_name(p) for p in players])

            for round in range(len(players)):
                players_instances = [p for p in players]
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

                # Rotating players for the next rounds
    #            initial_state.rotate_players()
                players.append(players.pop(0))
                results.append(results.pop(0))
        
        self.all_scores[0] += results[0]
        self.all_scores[1] += results[1]
        
        if results[0] > results[1]:
            self.score[0] += 1
            return 1
        if results[0] < results[1]:
            self.score[1] += 1
            return -1
        return 0
        
    def player_name(self, player):
        return player.__class__.__name__
    

models_dir = "models/PPO"
#models_dir = "DQNmodel/DQN"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if __name__ == '__main__':
    dir = "models/DQN"
    dir_path = f"{dir}/DQN.zip"
    env_lambda = lambda: CustomEnv()
    do_train = True
    Continue = False
    num_cpu = 8
    env = VecMonitor(SubprocVecEnv([env_lambda for i in range(num_cpu)]))

    if Continue and do_train:
        model_path = f"{models_dir}/rl_model_400000_steps"
        log_path = f"C:/Koodi/RL_AI/logs/PPO_2/"
        model = PPO.load(model_path, env=env, tensorboard_log=log_path)
        model.set_env(env)
        checkpoint_callback = CheckpointCallback(
            save_freq= 500,
            save_path=dir
        )
        model.learn(
            total_timesteps=20000, log_interval=1, reset_num_timesteps=False
        )
        model.save(f"{models_dir}/{2221}")

    elif do_train and not Continue:
        checkpoint_callback = CheckpointCallback(
            save_freq= 500,
            save_path=dir
        )
        """
        model = PPO(
            policy="MlpPolicy",
            env = env,
            verbose=1,
            tensorboard_log="./logs/",
            n_epochs=12,
            n_steps=512,
            device='cuda'
        )
        
        model = DQN(
            "MlpPolicy", 
            env, verbose=1,
            tensorboard_log="./logs/")
        """

        model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log="./logs/",
            device='cuda',
            learning_rate=0.0001,
        )
         
        
        model.learn(
            total_timesteps=500000,
            callback=[checkpoint_callback], log_interval=1
        )
        
        model.save(f"{models_dir}/{212}")
        '''TIMESTEPS = 500
        iters = 0
        for i in range(1,10):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback], tb_log_name='PPO')
            model.save(f"{models_dir}/{TIMESTEPS*i*num_cpu}")'''
    elif not do_train:
        episodes = 5
        model_path = f"{models_dir}/rl_model_400000_steps"
        log_path = f"C:/Koodi/RL_AI/logs/PPO_2/"
        model = PPO.load(model_path, env=env, tensorboard_log=log_path)
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, truncated, info = env.step(action)
                env.render()
    #model.save(f"{models_dir}/{num_cpu}")
    #model = PPO.load(f'{models_dir}/{num_cpu}.zip', env=env)
    exit()