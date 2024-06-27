
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv_6x6 import BattleModuleMagicTowerEnv_6x6
from Envs.modularEnv.CoinModuleMagicTowerEnv_6x6 import CoinModuleMagicTowerEnv_6x6
from Envs.modularEnv.KeyModuleMagicTowerEnv_6x6 import KeyModuleMagicTowerEnv_6x6
from Envs.modularEnv.ModuleMagicTowerEnv_6x6 import ModuleMagicTowerEnv_6x6
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
import numpy as np
import Data_processor
import train
from Developer_controller import Developer_controller
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
    Data_processor.Moudel_test(model,env,1000,100,1,ifprint = False,save_path ='logs/test_Log/more_battle_test.csv',developer_controller=developer_controller)
#BattleModuletrain()

env = make_vec_env("ModuleMagicTowerEnv_6x6")
model = RecurrentPPO.load('trained_modules/Controller/Controller_best')


#more_battle_developer_controller = Developer_controller([15,-30, -5])
#more_coin_developer_controller = Developer_controller([-15,60, 0])
#more_key_developer_controller = Developer_controller([-5,-10, 3])
#developer_controller = Developer_controller([-15,-30, 3])


Data_processor.Moudel_test(model,env,1000,100,save_path ='logs/test_Log/org_test_Log/')








#Data_processor.darw_state_value_map('logs/state_value_test_log/tate_value_test_log.csv','battle','mean','battle state value map(mean)')
#Data_processor.darw_state_value_map('logs/state_value_test_log/tate_value_test_log.csv','battle','max','battle state value map(max)')
#Data_processor.darw_state_value_map('logs/state_value_test_log/tate_value_test_log.csv','battle','min','battle state value map(min)')


#Data_processor.darw_track_map('logs/track_logs/org_track.csv','org track map')
#Data_processor.darw_track_map('logs/track_logs/battle_track.csv','battle track map')
#Data_processor.darw_track_map('logs/track_logs/noly_coin_track.csv','only key track map')