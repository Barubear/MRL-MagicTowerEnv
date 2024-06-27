
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def Moudel_test(model,env,test_times,max_step,save_path,developer_controller =None):
    state_value_list =[]
    log_list =[]
    track_list =[]
    for i in range(test_times):

        obs = env.reset()
        step =1
        info = None

        while True:

            if info == None:
                state_value_list.append([(1,5), obs['module_list'][0][0][1], obs['module_list'][0][1][1], obs['module_list'][0][2][1]])
                track_list.append((1,5))
            else:
                state_value_list.append([info[0]["pos"], obs['module_list'][0][0][1], obs['module_list'][0][1][1], obs['module_list'][0][2][1]])
                track_list.append(info[0]["pos"])
           

            if developer_controller != None:
                obs = developer_controller.add_weight(obs)

            action, _states = model.predict(obs)
            obs, rewards, dones, info  = env.step(action)
            step +=1
            if dones or step == max_step:
                print(i)
                log_list.append([step, info[0]["hp/enemy"][0], info[0]["hp/enemy"][1], info[0]["coin"]])

                break
            
    
    log_path = save_path+'test_log.csv'
    state_value_path = save_path+'state_value_log.csv'
    track_path = save_path+'trac_log.csv'

    write_log(log_path,log_list,['step','hp', 'enemy', 'coin'])
    write_log(state_value_path,state_value_list,['pos','battle', 'coin', 'key'])
    write_log(track_path,track_list)
    
def write_log(path,data,tile_list=None):
    with open(path, 'w',newline='') as f:
            writer = csv.writer(f)
            if tile_list != None:
                writer.writerow(tile_list)
            for msg in data:
                writer.writerow(msg)

def daw_graph(path1,path2,datatype,title=None,lable1 = 'org',lable2='new',xlable=None):
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
    hp_list2 = []
    enemy_list2= []
    coin_list2 =[]
    with open(path2,'r') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            hp_list2.append(row[1])
            enemy_list2.append(row[2])
            coin_list2.append(row[2])
            step_lsit2.append(row[0])


    log_dic ={}
    log_dic2 ={}
    log_list =[]
    log_list2 =[]
    if datatype == 'hp':
        log_list = hp_list
        log_list2= hp_list2
    elif datatype == 'enemy':
        log_list = enemy_list
        log_list2= enemy_list2
    elif datatype == 'coin':
        log_list = coin_list
        log_list2= coin_list2
    elif datatype == 'step':
        log_list = step_lsit
        log_list2= step_lsit2
    
    for n in log_list:
        
        if n in log_dic:
            log_dic[n] +=1
        else:
           log_dic[n] =1


    for n in log_list2:
        
        if n in log_dic2:
            log_dic2[n] +=1
        else:
           log_dic2[n] =1

    keys = list(set(list(log_dic.keys()) + list(log_dic2.keys())))
    keys.sort()
    
    values1 = [log_dic.get(key, 0) for key in keys]
    values2 = [log_dic2.get(key, 0) for key in keys]
    
    bar_width = 0.3
    r1 = range(len(keys))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, values1, color='r', width=bar_width, edgecolor='grey', label=lable1)

    plt.bar(r2, values2, color='b', width=bar_width, edgecolor='grey', label=lable2)
    
    if xlable !=None :
        plt.xlabel('coin')
    plt.ylabel('Count')
    if title !=None :
        plt.title(title)
    if  datatype != 'step':
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

def darw_state_value_map(path,modular,data_type,title= None):


    pos_dic = {}
    tuple_pos_dic ={}
    wall_pos = [(1,1),(4,1),(1,3),(3,3),(5,3),(1,4),(3,4),(5,4)]
    
    with open(path,'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pos = row[0]
            tuple_pos = tuple(map(int, row[0].strip('()').split(',')))
            if pos not in pos_dic:
                
                pos_dic[pos] = [[float(row[1])],[float(row[2])],[float(row[3])]]
                tuple_pos_dic[pos] = tuple_pos
            else:
                for i in range(3):
                    pos_dic[pos][i].append(float(row[i+1]))
             

    
    modular_index = -1
    if modular == "battle":
        modular_index = 0
    elif modular == "coin":
        modular_index = 1
    elif modular == "key":
        modular_index = 1
    else:
        return ("no modular named "+ modular)

    state_value_dic = [[0 for _ in range(6)] for _ in range(6)]
    state_value_list=[]

    for pos in pos_dic:

        value = 0
        tuple_pos = tuple_pos_dic[pos]
        if data_type == "mean":
            value =  np.array(pos_dic[pos][modular_index]).mean()
            state_value_list.append(value  )
            state_value_dic[tuple_pos[0]][tuple_pos[1]] = value
        elif data_type == "max":
            value =  np.array(pos_dic[pos][modular_index]).max()
            state_value_list.append(value  )
            state_value_dic[tuple_pos[0]][tuple_pos[1]] = value
        elif data_type == "min":
            value =  np.array(pos_dic[pos][modular_index]).min()
            state_value_list.append(value  )
            state_value_dic[tuple_pos[0]][tuple_pos[1]] = value
        elif data_type == "sub":
            value =  np.array(pos_dic[pos][modular_index]).max() - np.array(pos_dic[pos][modular_index]).min()
            state_value_list.append(value  )
            state_value_dic[tuple_pos[0]][tuple_pos[1]] = value
        else:
            return ("no data tpye named "+ modular)
        print(pos,value)

    state_value_list = np.array(state_value_list)

   
    

    # 创建一个颜色映射对象
    cmap = cm.get_cmap('viridis') 
    vmin = np.min(state_value_list) 
    vmax = np.max(state_value_list)  

    fig, ax = plt.subplots()

    # 绘制网格
    for i in range(6):
        for j in range(6):
            color= 'white'
            
                
            if (i,j) in wall_pos:
                color = 'black'
            else:
                color = cmap((state_value_dic[i][j] - vmin) / (vmax - vmin))
            rect = plt.Rectangle((i, j), 1, 1, facecolor=color)
            ax.add_patch(rect)

            
    # 设置坐标轴
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

# 显示绘图
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # 设置空数组，必须设置，否则报错
    plt.colorbar(sm, ax=ax)
    if title != None:
        plt.title(title)
    plt.gca().invert_yaxis()  # 将 y 轴反转以使 (0,0) 在左上角
    plt.show()

def darw_track_map(path,title=None ):

    track_map = [[0 for _ in range(6)] for _ in range(6)]
    with open(path,'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            pos_x,pos_y = int(row[0]),int(row[1])
            track_map[pos_x][pos_y] +=1
            

    color_list = []
    

    for i in range(6):
        for j in range(6):
            color_list.append(track_map[i][j])
    
    color_list = np.array(color_list) 

    cmap = cm.get_cmap('coolwarm') 
    vmin = np.min(color_list) 
    vmax = np.max(color_list) 

    fig, ax = plt.subplots()

    # 绘制网格
    for i in range(6):
        for j in range(6):
            color= 'white'
            
                
            
            color = cmap((track_map[i][j] - vmin) / (vmax - vmin))
            rect = plt.Rectangle((i, j), 1, 1, facecolor=color)
            ax.add_patch(rect)

            
    # 设置坐标轴
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

# 显示绘图
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # 设置空数组，必须设置，否则报错
    plt.colorbar(sm, ax=ax)
    if title != None:
        plt.title(title)
    plt.gca().invert_yaxis()  # 将 y 轴反转以使 (0,0) 在左上角
    plt.show()

def get_state_value(model,state,obs,device='cuda'):

    # 将obs字典中的每个值转换为PyTorch张量，并放入新的字典中
    obs_tensor_dict = {key: torch.as_tensor(obs, device=device) for (key, obs) in obs.items()}


    _states_tensor = torch.tensor(state,dtype=torch.float32).to(device)
    episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
    state_value = model.policy.predict_values(obs_tensor_dict,_states_tensor ,episode_starts)

    return state_value