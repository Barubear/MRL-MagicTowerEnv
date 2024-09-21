import torch
import numpy as np
import csv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from Envs.modularEnv.ModuleMagicTowerEnv_6x6_for_test import ModuleMagicTowerEnv_6x6_for_test
import os
import Developer_controller
from scipy.stats import pearsonr
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

def Moudel_test(env ,model,test_times:int,max_step:int,folder:str,dc):
        log_list =[]
        for i in range(test_times):

            obs = env.reset()
            step =1
            info = None
            n = 0
            action = None
            while True:
                
                if dc != None:
                    obs = add_weight(obs,dc)
                action, _states = model.predict(obs)
                obs, rewards, dones, info  = env.step(action)
                step +=1

                if dones or step == max_step:
                    print(i)
                    log_list.append([step, info[0]["hp/enemy"][0], info[0]["hp/enemy"][1], info[0]["coin"]])

                    break
                
                
                
                

        log_path =  folder+ '/'+ str(dc) + 'test_log.csv'
        write_log(log_path, log_list, ['step','hp', 'enemy', 'coin'] )
        
        return log_path


def add_weight(obs,dc):
        new_obs =obs.copy()

        for i in range(len(dc)):
            #print(new_obs['module_list'][0][i][1])
            new_obs['module_list'][0][i][1] += dc[i]
            new_obs["dc"]= np.array(dc, dtype=float)
        return new_obs

def write_log(path,data,tile_list=None,write_type ='w'):
        with open(path, write_type,newline='') as f:
                writer = csv.writer(f)
                if tile_list != None:
                    writer.writerow(tile_list)
                for msg in data:
                    writer.writerow(msg)

def read_data(path,datatype,max_value = 10**10,min_value =-1,list_data_type =int):
        log_list = []
        log_index = -1

        if datatype == 'hp' or datatype == 'enemy score':
            log_index = 1
        elif datatype == 'enemy' or datatype == 'coin score':
            log_index = 2
        elif datatype == 'coin' or datatype == 'step_score': 
            log_index = 3
        elif datatype == 'step':
            log_index = 0
        elif datatype == 'key score': 
            log_index = 4
        elif datatype == 'clear rate': 
            log_index = 5
        else:
            print('No datatype named' + datatype)
            return
        


        with open(path,'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:

                value = list_data_type(row[log_index])
                
                if value < max_value and value > min_value:
                    log_list.append(value)
        
        return log_list

def enemy_weighted_average(data):
       
        point_dic ={0:0,
                    1:0,
                    2:0,
                    3:0,
                    }
        for i in data:
            
            point_dic[i] +=1
        
        point = ( point_dic[0] + point_dic[1]+ 0.5 * point_dic [2] + 0.2 * point_dic[3] ) / len(data)
        return point

def coin_weighted_average(data):
       
        point_dic ={0:0,
                    1:0,
                    2:0,
                    3:0,
                    4:0,
                    }
        for i in data:
            
            point_dic[i] +=1
        
        point = ( point_dic[0] + 0.8*point_dic[1]+ 0.5 * point_dic [2] + 0.2 * point_dic[3] +  0.1 * point_dic[4]) / len(data)
        return point
    

def step_weighted_average(enemy_point,coin_point,data):
        if len(data) == 0:
            return 0
        point_dic ={0:0,
                    1:0,
                    2:0,
                    }
        for i in data:
            if i >= 100:
                continue

            elif i <=16 :
                point_dic[0] +=1
            elif i>16 and i <= 36:
                point_dic[1] +=1
            else :
                point_dic[2] +=1
            
        
        step_point = ( point_dic[0] + 0.5*point_dic[1]+ 0.2 * point_dic [2] ) / len(data)

        point = (step_point*3 - enemy_point - coin_point)/3
        return point,step_point

def get_clear_rate(data):
        clear_num = 0
        for i in data:
            if i <100:
                clear_num += 1
        return clear_num / len(data)

def get_score(file_path):

        
    enemy_log_list = read_data(file_path,'enemy')
    enemy_score = enemy_weighted_average(enemy_log_list)
    enemy_score = round(enemy_score, 3)
    coin_log_list = read_data(file_path,'coin')
    coin_score = coin_weighted_average(coin_log_list)
    coin_score = round(coin_score, 3)
    step_log_list = read_data(file_path,'step')
    
    key_score,step_score= step_weighted_average(enemy_score,coin_score,step_log_list)
    key_score = round(key_score, 3)
    step_score = round(step_score, 3)
    clear_rate = get_clear_rate(step_log_list)




    return[enemy_score,coin_score,step_score,step_score,clear_rate]


def main():
    
    score_log_list =[]
    module_path= 'trained_modules/Controller/new_Ctrl_best01'
    Ctrl_log_path = 'logs/new_Controller_Log'
    Ctrl_env = make_vec_env("ModuleMagicTowerEnv_6x6-v_test")
    model = RecurrentPPO.load(module_path)
    model.set_env(Ctrl_env)

    dc  = [0,0,0]
    print(str(dc))
    file_path  =  Ctrl_log_path+ '/'+ str(dc) + 'test_log.csv'
    score_list = get_score(file_path)
    score_log_list.append(score_list)


    
    for i in range(3):
            dc = [0,0,0]
            for j in range(21):
                dc_value = (j -10)
                if dc_value == 0:
                    continue
                dc[i] = dc_value
                print(str(dc))
                file_path = Ctrl_log_path+ '/'+ str(dc) + 'test_log.csv'
                score_list = get_score(file_path)
                score_list.append(str(dc))
                score_log_list.append(score_list)

    title = ['enemy score','coin score','step_score','key score','clear_rate','dc']
    score_log_list.sort(key=lambda x: x[0], reverse=True)
    save_path = 'newScore/enemy_core.csv'
    write_log(save_path,  score_log_list, title )

    score_log_list.sort(key=lambda x: x[1], reverse=True)
    save_path = 'newScore/coin_core.csv'
    write_log(save_path,  score_log_list, title )

    score_log_list.sort(key=lambda x: x[2], reverse=True)
    save_path = 'newScore/step_core.csv'
    write_log(save_path,  score_log_list, title )

    score_log_list.sort(key=lambda x: x[3], reverse=True)
    save_path = 'newScore/key_core.csv'
    write_log(save_path,  score_log_list, title )

    score_log_list.sort(key=lambda x: x[4], reverse=True)
    save_path = 'newScore/clear_rate.csv'
    write_log(save_path,  score_log_list, title )

    


main()