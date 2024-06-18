
import torch
import pygame
import sys
import  time
import numpy as np
import csv
import pprint



def Moudel_test(model,env,test_times = 10,max_step = 100,print_log_step = 1,ifprint = True,save_path = None):
    state_value_list =[]
    log_list =[]
    
    for i in range(test_times):
        obs = env.reset()
        over =False
        step =0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        while True:
        
            action, _states = model.predict(obs)

            device = torch.device('cuda' )  # 根据情况选择CUDA或CPU设备

        # 将obs字典中的每个值转换为PyTorch张量，并放入新的字典中
            obs_tensor_dict = {key: torch.as_tensor(obs, device=device) for (key, obs) in obs.items()}


            _states_tensor = torch.tensor(_states,dtype=torch.float32).to(device)
            episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
            state_value = model.policy.predict_values(obs_tensor_dict,_states_tensor ,episode_starts)
            state_value_list.append(state_value)
            obs, rewards, dones, info  = env.step(action)
            
            
            if ifprint and step % print_log_step == 0:
                print(info,action)
                print(state_value.item())
                print(obs['map'])
            

            if dones or step == max_step:
                print(step)
                
                
                log_list.append([step, info[0]["hp/enemy"][0], info[0]["hp/enemy"][1], info[0]["coin"]])

                break
            step +=1
    #state_value_list = [value.detach().cpu().numpy() for value in state_value_list]
    #np.array(state_value_list) ,
    if save_path == None:
        return np.array(log_list)
    else:
        with open(save_path, 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step','hp', 'enemy', 'coin'])
            for msg in log_list:
                writer.writerow(msg)
    



