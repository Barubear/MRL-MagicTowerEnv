
import torch
import pygame



def test(model,env,max_step = 100,print_log_step = 10):
    obs = env.reset()
    over =False
    step =0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while True:
        
        action, _states = model.predict(obs)
        obs_tensor = torch.tensor(obs).to(device)
        _states_tensor = torch.tensor(_states,dtype=torch.float32).to(device)
        episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
        state_value = model.policy.predict_values(obs_tensor,_states_tensor ,episode_starts)

        obs, rewards, dones, info  = env.step(action)
        
        
        

        if step % print_log_step == 0:
            print(state_value)
            print(info)
        if dones or step >=max_step:
            info = env.reset()
            break
        step +=1



