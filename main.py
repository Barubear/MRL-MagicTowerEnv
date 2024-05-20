
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
#lv1_env = make_vec_env("CurriculumMagicTowerEnv_lv1",monitor_dir="models")


#or_model = RecurrentPPO(
#    "MlpLstmPolicy",
#    vec_env,
#    device='cuda',
#    verbose=1
#)

lv1_path = 'CurriculumMdels/best_model_lv1'
lv2_path = 'CurriculumMdels/best_model_lv2'
#print(train.train(s_model,vec_env,3000000,lv1_path))
lv1_model = RecurrentPPO.load(lv1_path)
lv2_env = make_vec_env("CurriculumMagicTowerEnv_lv2",monitor_dir="models")
print(train.train(lv1_model,lv2_env ,3000000,lv2_path))
lv2_model = RecurrentPPO.load(lv2_path)
render_test.test(lv2_model,lv2_env,300,10) 

