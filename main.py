
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv_6x6 import BattleModuleMagicTowerEnv_6x6
from Envs.modularEnv.CoinModuleMagicTowerEnv_6x6 import CoinModuleMagicTowerEnv_6x6
from Envs.modularEnv.KeyModuleMagicTowerEnv_6x6 import KeyModuleMagicTowerEnv_6x6
from Envs.modularEnv.ModuleMagicTowerEnv_6x6 import ModuleMagicTowerEnv_6x6
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
import numpy as np
from Data_processor import Data_Processor 
import train
from Developer_controller import Developer_controller
import os

def Moduletrain():
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


env = make_vec_env("ModuleMagicTowerEnv_6x6")
model = RecurrentPPO.load('trained_modules/Controller/Controller_best')


img_save_path = 'D:/大学院/2024春/実装/実験記録/img'

more_battle_developer_controller_dic ={

"01" : Developer_controller([20,0, 0]),
"02" : Developer_controller([20,-40, -10]),
"03" : Developer_controller([30, 0, 0]),
"04" : Developer_controller([30,-40, -5]),
"05" : Developer_controller([35,-50, -6]),
"06" : Developer_controller([25,-50, -6]),
"07" : Developer_controller([20,-50, -5]),
"08" : Developer_controller([15,-30, -5]),
"09" : Developer_controller([15,-50, -5]),
"10" : Developer_controller([15, 0, 0]),
"11" : Developer_controller([10,-30, -5]),
"12" : Developer_controller([18,-30, -5]),
"13" : Developer_controller([13,-30, -5]),
"14" : Developer_controller([20,-30, -5]),

"15" : Developer_controller([14,-30, -5]),
"16" : Developer_controller([14,-40, -5]),
"17" : Developer_controller([14,-20, -5]),


}











def main():
    env = make_vec_env("ModuleMagicTowerEnv_6x6")
    model = RecurrentPPO.load('trained_modules/Controller/Controller_best')
    img_save_path = 'D:/大学院/2024春/実装/実験記録/img'
    
    dp = Data_Processor(env,model,'logs/test_log/','logs/test_Log/org_test_Log',img_save_path)
    dp.developer_controller_test('MoreBattle',
                                 more_battle_developer_controller_dic,
                                 list(more_battle_developer_controller_dic.keys()),
                                 only_draw = True,
                                 save_only= True)





main()