
import torch
import pygame
import sys
import  time

def next_step(model, obs,device):
    action, _states = model.predict(obs)
    obs_tensor = torch.tensor(obs).to(device)
    _states_tensor = torch.tensor(_states,dtype=torch.float32).to(device)
    episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
    state_value = model.policy.predict_values(obs_tensor,_states_tensor ,episode_starts)
    return action,state_value






def test(model,env,max_step = 100,print_log_step = 1,ifprint = True):
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

        obs, rewards, dones, info  = env.step(action)
        
        
        

        if ifprint and step % print_log_step == 0:
            print(info,action)
            print(state_value)
            print(obs['map'])
            

        if dones or step ==max_step:
            print(step)
            
            break
        step +=1



