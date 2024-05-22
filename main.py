
import gymnasium as gym
import MagicTowerEnv
import CurriculumMagicTowerEnv_lv1
import CurriculumMagicTowerEnv_lv1_with_winRate
import CurriculumMagicTowerEnv_lv2_with_winRate
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

def lv1_train():
    lv1_env = make_vec_env("CurriculumMagicTowerEnv_lv1_with_winRate",monitor_dir="models")


    or_model  = RecurrentPPO(
    "MlpLstmPolicy",
    lv1_env,
    learning_rate=3e-4,
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=1),
    verbose=1,
    
    )

    lv1_path = 'CurriculumMdels/best_model_lv1_with_winRate'

    #print(train.train(or_model,lv1_env,2000000,lv1_path))
    lv1_model = RecurrentPPO.load(lv1_path)
    render_test.test(lv1_model,lv1_env,500,10) 
#lv1_train()


def lv2_train():
    lv2_path = 'CurriculumMdels/best_model_lv2_with_winRate'
    lv1_path = 'CurriculumMdels/best_model_lv1_with_winRate'
    lv1_model = RecurrentPPO.load(lv1_path)
    lv2_env = make_vec_env("CurriculumMagicTowerEnv_lv2_with_winRate",monitor_dir="models")
    lv1_model.set_env(lv2_env)
    print(train.train(lv1_model,lv2_env ,3000000,lv2_path))
    lv2_model = RecurrentPPO.load(lv2_path)
    render_test.test(lv2_model,lv2_env,1000,10) 

lv2_train()
