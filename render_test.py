
import torch
import pygame
import sys
import  time
import numpy as np

def next_step(model, obs,device):
    action, _states = model.predict(obs)
    obs_tensor = torch.tensor(obs).to(device)
    _states_tensor = torch.tensor(_states,dtype=torch.float32).to(device)
    episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
    state_value = model.policy.predict_values(obs_tensor,_states_tensor ,episode_starts)
    return action,state_value






def test(model,env,test_times = 10,max_step = 100,print_log_step = 1,ifprint = True):
    state_value_list =[]
    step_list =[]
    hp_list =[]
    enemy_list =[]
    coin_list =[]
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
        
            hp_list.append(info['hp/enemy'][0])
            enemy_list.append(info['hp/enemy'][1])
            coin_list.append(info['coin'])

            if ifprint and step % print_log_step == 0:
                print(info,action)
                print(state_value.item())
                print(obs['map'])
            

            if dones or step ==max_step:
                print(step)
                step_list.append(step)
                hp_list.append(info['hp/enemy'][0])
                enemy_list.append(info['hp/enemy'][1])
                coin_list.append(info['coin'])
                break
            step +=1
    #state_value_list = [value.detach().cpu().numpy() for value in state_value_list]
    #np.array(state_value_list) ,
    return np.array(step_list),np.array(hp_list),np.array(enemy_list),np.array(coin_list)



