
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv import BattleModuleMagicTowerEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results,ts2xy
import sb3_contrib
from sb3_contrib import RecurrentPPO
import torch

import render_test
import train

def BattleModuletrain():
    save_path = 'modules/BattleModule/Battle_best_model'
    log_path = 'logs/BattleModuleLog'

    env = make_vec_env("BattleModuleMagicTowerEnv",monitor_dir=log_path)


    model  = RecurrentPPO(
    "MultiInputLstmPolicy",
    env,
    learning_rate=1e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.15,
    batch_size=512,
    n_steps=256,
    n_epochs=10,
    policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=2),
    verbose=1,
    
    )

    
   
    print(train.train(model,env,2000000,save_path,log_path,10))
    model = RecurrentPPO.load(save_path)
    render_test.test(model,env,1000,100) 
BattleModuletrain()


