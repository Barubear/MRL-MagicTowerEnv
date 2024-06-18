
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv_6x6 import BattleModuleMagicTowerEnv_6x6
from Envs.modularEnv.CoinModuleMagicTowerEnv_6x6 import CoinModuleMagicTowerEnv_6x6
from Envs.modularEnv.KeyModuleMagicTowerEnv_6x6 import KeyModuleMagicTowerEnv_6x6
from Envs.modularEnv.ModuleMagicTowerEnv_6x6 import ModuleMagicTowerEnv_6x6
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
import numpy as np
import render_test
import train

def BattleModuletrain():
    save_path = 'trained_modules/Controller/Controller_best'
    log_path = 'logs/Controller_Log'

    env = make_vec_env("ModuleMagicTowerEnv_6x6")#,monitor_dir=log_path


    model = RecurrentPPO(
    "MultiInputLstmPolicy",
    env,
    learning_rate=1e-4,  # 学习率
    gamma=0.995,  # 折扣因子
    gae_lambda=0.95,  # GAE λ
    clip_range=0.2,  # 剪辑范围
    ent_coef=0.15,  # 熵系数
    batch_size=512,  # 批大小
    n_steps=256,  # 步数
    n_epochs=16,  # 训练次数
    policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1),  # LSTM 设置
    verbose=1,
)

    
    #print(train.train(model,env,2000000,save_path,log_path,10))
    model = RecurrentPPO.load(save_path)
    
    state_value_list,step_list = render_test.test(model,env,1,100,1) 
    #print(np.mean(state_value_list))
    #print(np.max(state_value_list))
    #print(np.min(state_value_list))
    #print(np.mean(step_list))
    
BattleModuletrain()



