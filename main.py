
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
vec_env = make_vec_env("MagicTowerEnv-v0", n_envs=8,monitor_dir="models")
model = RecurrentPPO(
    "MlpLstmPolicy",
    vec_env,
    batch_size=1024,
    n_steps=512,  
    ent_coef=0.005,  
    clip_range=0.2,   
    n_epochs=8,
    gamma=0.995,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    vf_coef=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[512, 1024], vf=[512, 1024]),
        lstm_hidden_size=512,  # 内存大小
        n_lstm_layers=2
    ),
    device='cuda',
    verbose=1
)

print(train.train(model,vec_env,2000000))
model = RecurrentPPO.load("models/best_model" )
#render_test.render_test(model,vec_env,100) 
render_test.test(model,vec_env,200) 

