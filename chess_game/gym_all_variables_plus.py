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
from agents.random_agent import RandomAgent
from agents.DLAgent import DLAgent
from agents.testingAgent2 import TestingAgent2
from agents.DellAgent import DellAgent
from agents.Obstacle2 import Agent2
from agents.LenovoAgent import LenovoAgent
from agents.FishAgent import FishAgent
from agents.testingAgent import TestingAgent
from agents.custom_agent import CustomAgent
from agents.minimax_agent import MinimaxAgent
from chess_game.agents.Obstacle1 import Agent


from stable_baselines3 import PPO, A2C, DQN, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import SAC
from gym import spaces

NUM_CPU = 16

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
        
        self.agent = Agent()
        
        self.games_played = 0
        
        self.score = [0, 0]
        self.all_scores = [0,0]
        
        self.bishopweight = self.agent.bishopweight
        self.knightweight = self.agent.knightweight
        self.queenweight = self.agent.queenweight
        self.kingweight = self.agent.kingweight
        
        self.knight_attacking_value = self.agent.knight_attacking_value
        self.black_knight_attacking_value = self.agent.black_knight_attacking_value
        
        self.bishop_attacking_value = self.agent.bishop_attacking_value
        self.black_bishop_attacking_value = self.agent.black_bishop_attacking_value
        
        self.queen_attacking_value = self.agent.queen_attacking_value
        self.black_queen_attacking_value = self.agent.black_queen_attacking_value
        
        self.knight_pin_value = self.agent.knight_pinned_value
        self.bishop_pin_value = self.agent.bishop_pinned_value
        self.queen_pin_value = self.agent.queen_pinned_value
        
        self.num_tables = 4
        self.score = 0
        self.action_space = spaces.Box(low=-50, high=50, shape=(4 + 6 * 3 + 3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=20, shape=(2,1))


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
        expected_shape = (4 + 6 * 3 + 3,)
        if action.shape != expected_shape:
            print(f"Invalid action shape: {action.shape}, expected: {expected_shape}")
        
                
        for i in range(6):
            action_value = action[i]
            if i == 0:
                table = self.knight_attacking_value
            elif i == 1:
                table = self.black_knight_attacking_value
            elif i == 2:
                table = self.bishop_attacking_value
            elif i == 3:
                table = self.black_bishop_attacking_value
            elif i == 4:
                table = self.queen_attacking_value
            elif i == 5:
                table = self.black_queen_attacking_value
            for j in range(3):
                action_value = action[i * 3 + j]
                table[j] += action_value
                table = np.array(table)
        
        self.knight_pin_value += action[-6]
        self.bishop_pin_value += action[-5]
        self.queen_pin_value += action[-4]

        # Apply the last 4 actions to the weights
        self.bishopweight += action[-4]
        self.knightweight += action[-3]
        self.queenweight += action[-2]
        self.kingweight += action[-1]

        reward = self.calculate_reward()
        done = self.calculate_done()
        observation = np.array(self.all_scores, dtype=np.float32)
        observation = observation.reshape(-1, 1)
        truncated = False
        info = {"score": self.score, "games_played": self.games_played}

        return observation, reward, done, truncated, info


    
    def calculate_reward(self):
        # Play the game
        result = self.play_game()
        self.games_played += 1

        # Calculate the reward as the number of rounds won by the agent
        reward = (self.all_scores[0] - self.all_scores[1]) * 0.2
        return reward
    
    def reset(self, seed=None, options=None):
        # Reset the environment state here
        self.games_played = 0
        self.score = [0, 0]
        self.all_scores = [0, 0]

        # Initialize the observation with a meaningful value
        # For example, you can set the observation to a zero vector with the correct shape
        terminal_observation = np.array(self.all_scores, dtype=np.float32).reshape(-1, 1)
        info = {"terminal_observation": terminal_observation, "score": self.score, "games_played": self.games_played}
        return terminal_observation, info
    
    def close(self):
        return super().close()
    
    def calculate_done(self):
        if self.games_played >= 1:
            print(f"All scores: {self.all_scores}")
            print("\n")
            if self.all_scores[0] >= 13:
                print("Saving the tables to tables.json")
                with open('tables.json', 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:  # If the file is empty, set data to an empty list
                        data = []
                    data.append({
                        'score': self.score,
                        'all_scores': self.all_scores,
                        'bishopweight': self.bishopweight,
                        'knightweight': self.knightweight,
                        'queenweight': self.queenweight,
                        'kingweight': self.kingweight,
                        'knight_attacking_value': self.knight_attacking_value,
                        'black_knight_attacking_value': self.black_knight_attacking_value,
                        'bishop_attacking_value': self.bishop_attacking_value,
                        'black_bishop_attacking_value': self.black_bishop_attacking_value,
                        'queen_attacking_value': self.queen_attacking_value,
                        'black_queen_attacking_value': self.black_queen_attacking_value,
                        'knight_pin_value': self.knight_pin_value,
                        'bishop_pin_value': self.bishop_pin_value,
                        'queen_pin_value': self.queen_pin_value
                    })

                    with open('tables.json', 'w') as f:
                        json.dump(data, f)
            return True
        return False
    
    def play_game(self):
        ############### Set the players ###############
        self.agent.bishopweight = self.bishopweight
        self.agent.knightweight = self.knightweight
        self.agent.queenweight = self.queenweight
        self.agent.kingweight = self.kingweight
        
        self.agent.knight_attacking_value = self.knight_attacking_value
        self.agent.black_knight_attacking_value = self.black_knight_attacking_value
        self.agent.bishop_attacking_value = self.bishop_attacking_value
        self.agent.black_knight_attacking_value = self.black_knight_attacking_value
        self.agent.queen_attacking_value = self.queen_attacking_value
        self.agent.black_queen_attacking_value = self.black_queen_attacking_value
        
        self.agent.knight_pinned_value = self.knight_pin_value
        self.agent.bishop_pinned_value = self.bishop_pin_value
        self.agent.queen_pinned_value = self.queen_pin_value
        
        
        opponent = MinimaxAgent()
        players = [self.agent, opponent]

        results = [0, 0]
        for i in range(2):
            initial_state = State([self.player_name(p) for p in players])
            for round in range(len(players)):
                players_instances = [p for p in players]
                # Timeout for each move. Don't rely on the value of it. This
                # value might be changed during the tournament.
                timeouts = [2, 2]
                game = Game(players_instances)
                new_round = initial_state.clone()
                turn_duration_estimate = sum([t
                                            for p, t in zip(players, timeouts)
                                            if p != RandomAgent])
                winners = game.play(new_round,
                                    output=False,
                                    timeout_per_turn=timeouts)
                if len(winners) == 1:
                    results[winners[0]] += 1
                players.append(players.pop(0))
                results.append(results.pop(0))
        
        print(f"Game 1 played, results: {results}")
        
        if results[0] > results[1]:
            self.score[0] += 1
        elif results[0] < results[1]:
            self.score[1] += 1
        
        if results[1] >= 2:
            self.all_scores[0] += results[0]
            self.all_scores[1] += results[1]
            return -1
                
        opponent = MinimaxAgent()
        players = [self.agent, opponent]
        for i in range(2):
            initial_state = State([self.player_name(p) for p in players])
            for round in range(len(players)):
                players_instances = [p for p in players]
                # Timeout for each move. Don't rely on the value of it. This
                # value might be changed during the tournament.
                timeouts = [2, 2]
                game = Game(players_instances)
                new_round = initial_state.clone()
                turn_duration_estimate = sum([t
                                            for p, t in zip(players, timeouts)
                                            if p != RandomAgent])
                winners = game.play(new_round,
                                    output=False,
                                    timeout_per_turn=timeouts)
                if len(winners) == 1:
                    results[winners[0]] += 1
                players.append(players.pop(0))
                results.append(results.pop(0))
                
        print(f"Game 2 played, results: {results}")
        
        if results[0] > results[1]:
            self.score[0] += 1
        elif results[0] < results[1]:
            self.score[1] += 1
        
        if results[1] >=3:
            self.all_scores[0] += results[0]
            self.all_scores[1] += results[1]
            return -0.75
        
        opponent = MinimaxAgent()
        players = [self.agent, opponent]
        for i in range(2):
            initial_state = State([self.player_name(p) for p in players])
            for round in range(len(players)):
                players_instances = [p for p in players]
                # Timeout for each move. Don't rely on the value of it. This
                # value might be changed during the tournament.
                timeouts = [2, 2]
                game = Game(players_instances)
                new_round = initial_state.clone()
                turn_duration_estimate = sum([t
                                            for p, t in zip(players, timeouts)
                                            if p != RandomAgent])
                winners = game.play(new_round,
                                    output=False,
                                    timeout_per_turn=timeouts)
                if len(winners) == 1:
                    results[winners[0]] += 1
                players.append(players.pop(0))
                results.append(results.pop(0))
        
        print(f"Game 3 played, results: {results}")
        
        if results[0] > results[1]:
            self.score[0] += 1
        elif results[0] < results[1]:
            self.score[1] += 1
        
        if results[1] >= 4:
            self.all_scores[0] += results[0]
            self.all_scores[1] += results[1]
            return 0.2
        
        opponent = MinimaxAgent()
        players = [self.agent, opponent]
        for i in range(2):
            initial_state = State([self.player_name(p) for p in players])
            for round in range(len(players)):
                players_instances = [p for p in players]
                timeouts = [2, 2]
                game = Game(players_instances)
                new_round = initial_state.clone()
                turn_duration_estimate = sum([t
                                            for p, t in zip(players, timeouts)
                                            if p != RandomAgent])
                winners = game.play(new_round,
                                    output=False,
                                    timeout_per_turn=timeouts)
                if len(winners) == 1:
                    results[winners[0]] += 1
                players.append(players.pop(0))
                results.append(results.pop(0))
                
        if results[0] > results[1]:
            self.score[0] += 1
        elif results[0] < results[1]:
            self.score[1] += 1
            
        print(f"Game 4 played, results: {results}")
        
        if results[1] >= 5:
            self.all_scores[0] += results[0]
            self.all_scores[1] += results[1]
            return 0.6
        
        opponent = MinimaxAgent()
        players = [self.agent, opponent]
        for i in range(2):
            initial_state = State([self.player_name(p) for p in players])
            for round in range(len(players)):
                players_instances = [p for p in players]
                timeouts = [2, 2]
                game = Game(players_instances)
                new_round = initial_state.clone()
                turn_duration_estimate = sum([t
                                            for p, t in zip(players, timeouts)
                                            if p != RandomAgent])
                winners = game.play(new_round,
                                    output=False,
                                    timeout_per_turn=timeouts)
                if len(winners) == 1:
                    results[winners[0]] += 1
                players.append(players.pop(0))
                results.append(results.pop(0))
        
        print(f"Game 5 played, results: {results}")
        
        if results[0] > results[1]:
            self.score[0] += 1
        elif results[0] < results[1]:
            self.score[1] += 1
        
        if results[1] >= 6:
            self.all_scores[0] += results[0]
            self.all_scores[1] += results[1]
            return 0.5
        
        self.all_scores[0] += results[0]
        self.all_scores[1] += results[1]
        
        return 1
        
    def player_name(self, player):
        return player.__class__.__name__
    

models_dir = "models/SAC"
#models_dir = "DQNmodel/DQN"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if __name__ == '__main__':
    dir = "models/SAC"
    dir_path = f"{dir}/SAC.zip"
    env_lambda = lambda: CustomEnv()
    do_train = True
    Continue = False
    num_cpu = NUM_CPU
    env = VecMonitor(SubprocVecEnv([env_lambda for i in range(num_cpu)]))
    """
    if Continue and do_train:
        model_path = f"{models_dir}/rl_model_400000_steps"
        log_path = f"C:/Koodi/RL_AI/logs/SAC/"
        model = PPO.load(model_path, env=env, tensorboard_log=log_path)
        model.set_env(env)
        checkpoint_callback = CheckpointCallback(
            save_freq= 16,
            save_path=dir
        )
        model.learn(
            total_timesteps=20000, log_interval=1, reset_num_timesteps=False
        )
        model.save(f"{models_dir}/{2221}")
        
    """

    if do_train and not Continue:
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
        """
        
        
        model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log="./logs/",
            device='cuda',
            learning_rate=0.0005,
            learning_starts=64
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq= 200,
            save_path=dir,
            name_prefix='rl_model'
        )
         
        
        model.learn(
            total_timesteps=50000, log_interval=1, callback=checkpoint_callback
        )
        
        model.save(f"{models_dir}/{212}")
        '''TIMESTEPS = 500
        iters = 0
        for i in range(1,10):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback], tb_log_name='PPO')
            model.save(f"{models_dir}/{TIMESTEPS*i*num_cpu}")'''

    exit()