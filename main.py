
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
developer_controller = Developer_controller([-15,-30, 3])
Data_processor.Moudel_test(model,env,10000,100,1,ifprint = False,save_path ='logs/test_Log/only_key_test.csv',developer_controller=developer_controller)
#Data_processor.Moudel_test(model,env,10000,100,1,ifprint = False,save_path ='logs/test_Log/org_test.csv')
#Data_processor.daw_graph('logs/test_Log/org_test.csv','logs/test_Log/more_key_test.csv')

mean_step,mean_enemy,mean_coin = Data_processor.print_data('logs/test_Log/org_test.csv','logs/test_Log/only_key_test.csv')
print(mean_step)
print(mean_enemy)
print(mean_coin)

