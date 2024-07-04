
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv_6x6 import BattleModuleMagicTowerEnv_6x6
from Envs.modularEnv.CoinModuleMagicTowerEnv_6x6 import CoinModuleMagicTowerEnv_6x6
from Envs.modularEnv.KeyModuleMagicTowerEnv_6x6 import KeyModuleMagicTowerEnv_6x6
from Envs.modularEnv.ModuleMagicTowerEnv_6x6 import ModuleMagicTowerEnv_6x6
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
from Data_processor import Data_Processor 
import train
from Developer_controller import Developer_controller


def Moduletrain(save_path,log_path,env):
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
    
    train.train(model,env,3000000,save_path,log_path,100)





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

"15" : Developer_controller([15,-40, -5]),
"16" : Developer_controller([15,-20, -5]),

"17" : Developer_controller([-10,0, 0]),
"18" : Developer_controller([-15, 0, 0]),
"19" : Developer_controller([-20, 0, 0]),
"20" : Developer_controller([-5, 0, 0]),

"21" : Developer_controller([-30,0, 0]),
"22" : Developer_controller([-40, 0, 0]),
"23" : Developer_controller([-50, 0, 0]),
"24" : Developer_controller([10, 0, 0]),
"25" : Developer_controller([25, 0, 0]),

"26" : Developer_controller([13, 0, 0]),
"27" : Developer_controller([8, 0, 0]),
"28" : Developer_controller([5, 0, 0]),
}


coin_developer_controller_dic ={
    "01" : Developer_controller([-15,100, -5]),
    "02" : Developer_controller([-15,-60, -5]),
    "03" : Developer_controller([0,100, 0]),
    "04" : Developer_controller([0,-60, 0]),


    "05" : Developer_controller([0,120, 0]),
    "06" : Developer_controller([0,90, 0]),
    "07" : Developer_controller([0,-100, 0]),
    "08" : Developer_controller([0,-120, 0]),

    "09" : Developer_controller([0,80, 0]),
    "10" : Developer_controller([0,60, 0]),
    "11" : Developer_controller([0,45, 0]),
    "12" : Developer_controller([0,30, 0]),

    "13" : Developer_controller([0,50, 0]),

    "14" : Developer_controller([0,-50, 0]),

}


key_developer_controller_dic ={
    "01" : Developer_controller([0,0, 5]),
    "02" : Developer_controller([0,0, 10]),
    "03" : Developer_controller([0,0, 15]),

    
    "04" : Developer_controller([0,0, -10]),
    "05" : Developer_controller([0,0, -20]),
    "06" : Developer_controller([0,0, -50]),

    "07" : Developer_controller([0,0, 1]),
    "08" : Developer_controller([0,0, 15]),
    "09" : Developer_controller([0,0, 20]),

    "10" : Developer_controller([0,0, -1]),
    "11" : Developer_controller([0,0, -5]),
    "12" : Developer_controller([0,0, 50]),

    
    "13" : Developer_controller([0,0, -70]),
    "14" : Developer_controller([0,0, -30]),

    "15" : Developer_controller([0,0, -90]),
    "16" : Developer_controller([0,0, -110]),

    "17" : Developer_controller([0,0, -60]),
    "18" : Developer_controller([0,0, -40]),
    
}



def Battle_train():
    Battle_env = make_vec_env("BattleModuleMagicTowerEnv_6x6")
    pass

def Coin_train():
    Coin_env = make_vec_env("CoinModuleMagicTowerEnv_6x6")
    pass

def Key_train():
    Key_save_path = 'trained_modules/KeyModule/Key_best03'
    Key_log_path = 'logs/Key03_Log'
    pass

def Ctrl_train():
    Ctrl_save_path= 'trained_modules/Controller/Ctrl_best02'
    Ctrl_log_path = 'logs/Controller02_Log'
    Ctrl_env = make_vec_env("ModuleMagicTowerEnv_6x6",monitor_dir=Ctrl_log_path)
    Moduletrain(Ctrl_save_path,Ctrl_log_path,Ctrl_env)

def def_DP():
    env = make_vec_env("ModuleMagicTowerEnv_6x6")
    model = RecurrentPPO.load('trained_modules/Controller/Ctrl_best02')
    #img_save_path = 'D:/大学院/2024春/実装/実験記録/img'
    img_save_path_round2 = 'D:/大学院/2024春/実装/実験記録/img02'
    dp = Data_Processor(env,model,'logs/test_log_round2/','logs/test_Log/org_test_Log',img_save_path_round2)

    return dp

def get_score(dp):
    dp.developer_controller_test('MoreBattle',
                                 more_battle_developer_controller_dic,
                                 list(more_battle_developer_controller_dic.keys()),
                                 #only_draw = True,
                                 save_only= True
                                 )
    
    
    dp.developer_controller_test('MoreCoin',
                                 coin_developer_controller_dic,
                                 list(coin_developer_controller_dic.keys()),
                                 only_draw = True,
                                 save_only= True
                                 )
    
    
    dp.developer_controller_test('MoreKey',key_developer_controller_dic,
                                list(key_developer_controller_dic.keys()),
                                #only_draw = True,
                                save_only= True
                                )
    
    
    #dp.print_state_vale()
    #dp.get_score('MoreBattle',more_battle_developer_controller_dic,list(more_battle_developer_controller_dic.keys()))
    #dp.get_score('MoreCoin',coin_developer_controller_dic,list(coin_developer_controller_dic.keys()))
    #dp.get_score('MoreKey',key_developer_controller_dic,list(key_developer_controller_dic.keys()))

    #
    #print(dp.get_one_score('logs/test_Log/org_test_Log/test_log.csv'))
    #dp.get_pearsonr('MoreBattle','enemy score','coin score')
    

    
    #dp.get_score('MoreBattle',more_battle_developer_controller_dic,list(more_battle_developer_controller_dic.keys()))
    

def main():

    
    Ctrl_train()
    

    


main()