
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
model = RecurrentPPO(
    "MlpLstmPolicy",
    vec_env,
    batch_size=1024,
    n_steps=10240 // 1024,  # 等价于 buffer_size / batch_size
    learning_rate=0.0001,
    ent_coef=0.005,  # 类似于beta
    clip_range=0.2,  # 类似于epsilon
    n_epochs=3,
    gamma=0.995,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    vf_coef=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[1024, 1024, 1024, 1024], vf=[1024, 1024, 1024, 1024]),
        lstm_hidden_size=256,  # 内存大小
        n_lstm_layers=3
    ),
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=1
)

print(train.train(model,vec_env,3000000))
model = RecurrentPPO.load("models/best_model" )
#render_test.render_test(model,vec_env,100) 
render_test.test(model,vec_env,200) 

