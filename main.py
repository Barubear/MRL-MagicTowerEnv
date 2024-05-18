
import gymnasium as gym
import MagicTowerEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import sb3_contrib
from sb3_contrib import RecurrentPPO
import torch

import render_test
import train


vec_env = make_vec_env("MagicTowerEnv-v0",monitor_dir="models")
print(train.train(vec_env,5000000))
model = RecurrentPPO.load("models/best_model" )
#render_test.render_test(model,vec_env,100) 
render_test.test(model,vec_env,100) 

