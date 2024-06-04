
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv_6x6 import BattleModuleMagicTowerEnv_6x6
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

    env = make_vec_env("BattleModuleMagicTowerEnv_6x6",monitor_dir=log_path)


    model = RecurrentPPO(
    "MultiInputLstmPolicy",
    env,
    verbose=1,
    
    ) 

    print(train.train(model,env,3000000,save_path,log_path,10))
    model = RecurrentPPO.load(save_path)
    render_test.test(model,env,500,10) 
BattleModuletrain()



