
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt

def Moudel_test(model,env,test_times = 10,max_step = 100,print_log_step = 1,ifprint = True,save_path = None,developer_controller =None):
    state_value_list =[]
    log_list =[]
    
    for i in range(test_times):
        obs = env.reset()
        over =False
        step =0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        while True:
        
            if developer_controller != None:
                obs = developer_controller.add_weight(obs)

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
                print(i,step)
                
                
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
    


def daw_graph(path1,path2):
    step_lsit = []
    hp_list = []
    enemy_list= []
    coin_list =[]
    
    with open(path1,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            #step_lsit.append(row[0])
            #hp_list.append(row[1])
            #enemy_list.append(row[2])
            coin_list.append(row[3])

    step_lsit2 = []
    hp_list2 = []
    enemy_list2= []
    coin_list2 =[]
    with open(path2,'r') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            coin_list2.append(row[2])



    hp_dic ={}
    hp_dic2 ={}
    for n in coin_list:
        
        if n in hp_dic:
            hp_dic[n] +=1
        else:
            hp_dic[n] =1


    for n in coin_list2:
        
        if n in hp_dic2:
            hp_dic2[n] +=1
        else:
            hp_dic2[n] =1

    keys = list(set(list(hp_dic.keys()) + list(hp_dic2.keys())))
    keys.sort()
    
    values1 = [hp_dic.get(key, 0) for key in keys]
    values2 = [hp_dic2.get(key, 0) for key in keys]
    
    bar_width = 0.3
    r1 = range(len(keys))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, values1, color='r', width=bar_width, edgecolor='grey', label='org')
    plt.bar(r2, values2, color='b', width=bar_width, edgecolor='grey', label='battle')
    
    plt.xlabel('Coin')
    plt.ylabel('Count')
    plt.title('Coin Distribution')
    plt.xticks([r + bar_width/2 for r in range(len(keys))], keys)
    plt.legend()
    plt.show()

def print_data(path1,path2):
    step_lsit = []
    hp_list = []
    enemy_list= []
    coin_list =[]
    
    with open(path1,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            step_lsit.append(row[0])
            hp_list.append(row[1])
            enemy_list.append(row[2])
            coin_list.append(row[3])

    step_lsit2 = []
    hp_list2 =[]
    enemy_list2= []
    coin_list2 =[]
    with open(path2,'r') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            step_lsit2.append(row[0])
            hp_list2.append(row[1])
            enemy_list2.append(row[2])
            coin_list2.append(row[2])
    
    step_lsit = np.array(step_lsit,dtype= int)
    step_lsit2 = np.array(step_lsit2,dtype= int)
    enemy_list = np.array(enemy_list,dtype= int)
    enemy_list2 = np.array(enemy_list2,dtype= int)
    coin_list = np.array(coin_list,dtype= int)
    coin_list2 = np.array(coin_list2,dtype= int)
    
    mean_step = (np.mean(step_lsit),np.mean(step_lsit2))
    mean_enemy = (np.mean(enemy_list),np.mean(enemy_list2))
    mean_coin = (np.mean(coin_list),np.mean(coin_list2))
    return  mean_step,mean_enemy,mean_coin