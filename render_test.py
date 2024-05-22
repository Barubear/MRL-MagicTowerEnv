
import torch
import pygame
import sys
import  time
def render_test(model,env,max_step = 1000):
    obs = env.reset()
    over =False
    step =0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dones  =False

    pygame.init() 
    canvas =pygame.display.set_mode((900,900))
    pygame.display.set_caption("MagicTowerEnv")
    pix_square_size = 10
    
    canvas.fill((255, 255, 255))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        action,state_value = next_step(model, obs,device)
        data_map = obs[0]
        if step <= max_step and dones ==False:
            for x in range(0,6):
                for y in range(0,7):
                    color =(255,255,255)
                    if(data_map[x][y] == -1):
                     color = (0,0,0)
                    if(data_map[x][y] >=10 or data_map[x][y] <= -10):
                     color = (0,0,255)
                    if(data_map[x][y] == 3):
                     color = (0,255,0)
                    if(data_map[x][y] == 2):
                     color = (255,0,0)
                    if(data_map[x][y] == 4):
                     color = (255,255,0)  
                    if(data_map[x][y] == 5):
                     color = (255,160,0) 

                    pos = (x*pix_square_size,y *pix_square_size,100 ,100)
                    pygame.draw.rect(canvas,color,pos,0)

            obs, rewards, dones, info  = env.step(action)
            step += 1
        time.sleep(3)

def next_step(model, obs,device):
    action, _states = model.predict(obs)
    obs_tensor = torch.tensor(obs).to(device)
    _states_tensor = torch.tensor(_states,dtype=torch.float32).to(device)
    episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
    state_value = model.policy.predict_values(obs_tensor,_states_tensor ,episode_starts)
    return action,state_value






def test(model,env,max_step = 100,print_log_step = 1):
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
            print(info,action)
            print(state_value)
            print(obs)

        if dones or step ==max_step:
            print(step)
            
            break
        step +=1



