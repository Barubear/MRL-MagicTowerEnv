
import gymnasium as gym
from Envs.modularEnv.BattleModuleMagicTowerEnv_6x6 import BattleModuleMagicTowerEnv_6x6
from Envs.modularEnv.CoinModuleMagicTowerEnv_6x6 import CoinModuleMagicTowerEnv_6x6
from Envs.modularEnv.KeyModuleMagicTowerEnv_6x6 import KeyModuleMagicTowerEnv_6x6
from Envs.modularEnv.ModuleMagicTowerEnv_6x6 import ModuleMagicTowerEnv_6x6
from Envs.modularEnv.OldModuleMagicTowerEnv_6x6 import OldModuleMagicTowerEnv_6x6 
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
from Data_processor import Data_Processor 
import train
from Developer_controller import Developer_controller


def Moduletrain(save_path,log_path,env,times = 2000000):
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
    
    train.train(model,env,times,save_path,log_path,100)





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
    "15" : Developer_controller([-15,60, 0]),

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

    "19" : Developer_controller([-15,-30, 5]),
    "20" : Developer_controller([15,-30, 5]),
    "21" : Developer_controller([15,-30, 1]),
    "22" : Developer_controller([-15,-60, 0]),
    "23" : Developer_controller([5,0, -50]),
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
    Ctrl_env = make_vec_env("KeyModuleMagicTowerEnv_6x6",monitor_dir=Key_log_path)
    Moduletrain(Key_save_path,Key_log_path,Ctrl_env,1000000)
    

def Ctrl_train():
    Ctrl_save_path= 'trained_modules/Controller/Ctrl_best02'
    Ctrl_log_path = 'logs/Controller02_Log'
    Ctrl_env = make_vec_env("ModuleMagicTowerEnv_6x6",monitor_dir=Ctrl_log_path)
    Moduletrain(Ctrl_save_path,Ctrl_log_path,Ctrl_env,3000000)

def old_Ctrl_train():
    Ctrl_save_path= 'trained_modules/OldController/Ctrl_best02'
    Ctrl_log_path = 'logs/OldController_Log'
    Ctrl_env = make_vec_env("OldModuleMagicTowerEnv_6x6",monitor_dir=Ctrl_log_path)
    Moduletrain(Ctrl_save_path,Ctrl_log_path,Ctrl_env,3000000)
    #state_value:
    #max:23.179083
    #mean:-8.629469
    #min:-165.52904

def def_DP():
    env = make_vec_env("ModuleMagicTowerEnv_6x6")
    model = RecurrentPPO.load('trained_modules/Controller/Ctrl_best02')
    img_save_path = 'D:/大学院/2024春/実装/実験記録/img'
    img_save_path_round2 = 'D:/大学院/2024春/実装/実験記録/img02'
    dp = Data_Processor(env,model,'logs/test_log/','logs/test_Log/org_test_Log',img_save_path)

    return dp

def get_score(dp:Data_Processor):
    """
    dp.developer_controller_test('MoreBattle',
                                 more_battle_developer_controller_dic,
                                 list(more_battle_developer_controller_dic.keys()),
                                 
                                 save_only= True,
                                 
                                 )
    
    
    dp.developer_controller_test('MoreCoin',
                                 coin_developer_controller_dic,
                                 ["15" ],
                                 
                                 save_only= True,
                                 )
               """                  
    
    
    dp.developer_controller_test('MoreKey',key_developer_controller_dic,
                                ["23"],
                                
                                save_only= True,
                                
                                )
    
    
    
    #dp.get_score('MoreBattle',more_battle_developer_controller_dic,list(more_battle_developer_controller_dic.keys()))
    #dp.get_score('MoreCoin',coin_developer_controller_dic,list(coin_developer_controller_dic.keys()))
    dp.get_score('MoreKey',key_developer_controller_dic,list(key_developer_controller_dic.keys()))

    
    #print(dp.get_one_score('logs/test_Log_round2/org_test_Log/test_log.csv'))
    
    

def main():

    
    #Ctrl_train()
    #Key_train()
    #env = make_vec_env("ModuleMagicTowerEnv_6x6")
    #model = RecurrentPPO.load('trained_modules/Controller/Ctrl_best02.zip')
    dp = def_DP()
    dp.Moudel_test(1000,100,'logs/test_log/org_test_Log')
    #dp.darw_track_map('logs/test_log_round2/org_test_Log/trac_log.csv','org track map','D:/大学院/2024春/実装/実験記録/img02/org')
    """
    dp.daw_graph('enemy','logs/test_Log/org_test_Log/test_log.csv' ,title=' org enemy count',xlable='enemy',img_save_path = 'D:/大学院/2024春/実装/実験記録/img/org',save_only =True)
    dp.daw_graph('coin','logs/test_Log/org_test_Log/test_log.csv',title= 'org coin count',xlable='coin',img_save_path = 'D:/大学院/2024春/実装/実験記録/img/org',save_only =True)
    dp.daw_graph('step','logs/test_Log/org_test_Log/test_log.csv', title=' org step count',xlable='step',img_save_path = 'D:/大学院/2024春/実装/実験記録/img/org',save_only =True)
    dp.daw_graph('hp','logs/test_Log/org_test_Log/test_log.csv', title=' org hp count',xlable='step',img_save_path = 'D:/大学院/2024春/実装/実験記録/img/org',save_only =True)
    
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"battle","max",'org battle max')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"battle","mean",'org battle mean')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"battle","min",'org battle min')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"coin","max",'org coin max')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"coin","mean",'org coin mean')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"coin","min",'org coin min')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"key","max",'org key max')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"key","mean",'org key mean')
    dp.darw_state_value_map('logs/test_log_round2/org_test_Log/state_value_log.csv',"key","min",'org key min')
    """
    #get_score(dp)
    #dp.daw_graph('step','logs/test_log/org_test_Log/test_log.csv','logs/test_Log/MoreBattle_test25_Log/test_log.csv',title='MoreBattle_test25 step count',lable1 = 'org',lable2='moreBattle25',xlable='step')
   # dp.print_state_vale('logs/test_log_round2/org_test_Log/state_value_log.csv')

main()