
import gymnasium as gym
import MagicTowerEnv
import CurriculumMagicTowerEnv_lv1
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
vec_env = make_vec_env("CurriculumMagicTowerEnv_lv1",monitor_dir="models")
model = RecurrentPPO(
    "MlpLstmPolicy",
    vec_env,
    batch_size=1024,
    n_steps=128,  
    ent_coef=0.005,  
    clip_range=0.3,   
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    learning_rate = 0.0002,
    max_grad_norm=0.7,
    vf_coef=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[512, 1024,1024], vf=[512, 1024,1024]),
        lstm_hidden_size=512,  
        n_lstm_layers=2
    ),
    device='cuda',
    verbose=1
)

s_model = RecurrentPPO(
    "MlpLstmPolicy",
    vec_env,
    device='cuda',
    verbose=1
)
lv1_path = 'CurriculumMdels/best_model_lv1'
print(train.train(s_model,vec_env,3000000,lv1_path))
s_model = RecurrentPPO.load(lv1_path )
#render_test.render_test(model,vec_env,100) 
render_test.test(s_model,vec_env,200) 

