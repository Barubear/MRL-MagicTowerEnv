
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv import BattleModuleMagicTowerEnv
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

def BattleModuletrain():
    env = make_vec_env("BattleModuleMagicTowerEnv",monitor_dir="BattleModules")


    model  = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.1,
    n_steps=32,
    batch_size=64,
    n_epochs=6,
    policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=2),
    verbose=1,
    
    )

    save_path = 'CurriculumMdels_Round3/best_model_lv1_reTrain_with_winRate'
    
   
    print(train.train(model,env,3000000,save_path))
    model = RecurrentPPO.load(save_path)
    render_test.test(model,env,1000,20) 
BattleModuletrain()


