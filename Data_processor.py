
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


class Data_Processor:
    def __init__(self,env,model,log_save_path,org_log_path = None,img_save_path = None) :
        self.env = env
        self.model = model
        self.log_save_path = log_save_path
        self.org_log_path = org_log_path
        self.img_save_path =img_save_path
        
        pass



    def get_state_value(self,state,obs,device='cuda'):

    # 将obs字典中的每个值转换为PyTorch张量，并放入新的字典中
        obs_tensor_dict = {key: torch.as_tensor(obs, device=device) for (key, obs) in obs.items()}


        _states_tensor = torch.tensor(state,dtype=torch.float32).to(device)
        episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
        state_value = self.model.policy.predict_values(obs_tensor_dict,_states_tensor ,episode_starts)
        return state_value

    def Moudel_test(self,test_times,max_step,folder,developer_controller =None):
        state_value_list =[]
        log_list =[]
        track_list =[]
        for i in range(test_times):

            obs = self.env.reset()
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

                action, _states = self.model.predict(obs)
                obs, rewards, dones, info  = self.env.step(action)
                step +=1
                if dones or step == max_step:
                    print(i)
                    log_list.append([step, info[0]["hp/enemy"][0], info[0]["hp/enemy"][1], info[0]["coin"]])

                    break
            
    
            log_path =  folder +'/test_log.csv'
            state_value_path =  folder +'/state_value_log.csv'
            track_path =  folder +'/trac_log.csv'

        self.write_log(log_path, log_list, ['step','hp', 'enemy', 'coin'] )
        self.write_log(state_value_path, state_value_list, ['pos','battle', 'coin', 'key'])
        self.write_log(track_path, track_list)
    
    def write_log(self,path,data,tile_list=None):
        with open(path, 'w',newline='') as f:
                writer = csv.writer(f)
                if tile_list != None:
                    writer.writerow(tile_list)
                for msg in data:
                    writer.writerow(msg)

    def daw_graph(self,datatype,path1= None,path2 =None,title=None,lable1 = 'org',lable2='new',xlable=None,img_save_path = None,save_only = False):
        step_lsit = []
        hp_list = []
        enemy_list= []
        coin_list =[]

        if path1 == None:
            path1 = self.org_log_path

        with open(path1,'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                step_value = int(row[0])
                if step_value < 100:
                    step_lsit.append(int(row[0]))
                hp_list.append(int(row[1]))
                enemy_list.append(int(row[2]))
                coin_list.append(int(row[3]))
    
        step_lsit2 = []
        hp_list2 = []
        enemy_list2= []
        coin_list2 =[]
        if path2 !=None:
            with open(path2,'r') as f2:
                reader = csv.reader(f2)
                next(reader)
                for row in reader:
                    step_value = int(row[0])
                    if step_value < 100:
                        step_lsit2.append(int(row[0]))
                    hp_list2.append(int(row[1]))
                    enemy_list2.append(int(row[2]))
                    coin_list2.append(int(row[3]))
                


    
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
        else:
            print('No datatype named' + datatype)
            return
    
        log_dic ={}
        log_dic2 ={}
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
        if datatype == 'step':
            plt.figure(figsize=(15,4.8))

        plt.bar( r1, values1, color='r', width=bar_width, edgecolor='grey', label=lable1)

        if path2 != None:
            plt.bar(  r2, values2, color='b', width=bar_width, edgecolor='grey', label=lable2)
    
        if xlable !=None :
            plt.xlabel(xlable)

        plt.ylabel('Count')
        if title !=None :
            plt.title(title)
        if  datatype != 'step' or path2 != None:
            plt.xticks([r + bar_width/2 for r in range(len(keys))], keys)

        if datatype == 'step' :
            
            xticks_labels = [key if i % 10 == 0 else '' for i, key in enumerate(keys)]
            plt.xticks([r + bar_width / 2 for r in range(len(keys))], xticks_labels)

        plt.legend()

        for i in range(len(r1)):
            plt.text(r1[i], values1[i] + 0.5, str(values1[i]), ha='center', va='bottom', fontsize=6)
            if path2 is not None:
                plt.text(r2[i], values2[i] + 0.5, str(values2[i]), ha='center', va='bottom', fontsize=6)

        if img_save_path != None:
            plt.savefig(img_save_path+'/'+ title + '.png', format='png', dpi=600)

        if save_only == False:
            plt.show()
        else:
            plt.close()

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
            modular_index = 2
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

    def darw_track_map(self,path,title=None,img_save_path = None ,save_only = False):

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
        if img_save_path != None:
            plt.savefig(img_save_path+'/'+ title + '.png', format='png', dpi=600)
        if not save_only:

            plt.show()
        else:
            plt.close()

    def developer_controller_test(self,moduar_name,dc_dic,dc_index_list,only_draw = False,save_only = False):
        
        directory_path_list =[]
        img_directory_path_list =[]
        i= 0
        for dc_keys in dc_index_list:
            
            directory_path = self.log_save_path+moduar_name+'_test'+ dc_keys + '_Log'
            directory_path_list.append(directory_path)
            os.makedirs(directory_path, exist_ok=True)
            if only_draw == False:
                self.Moudel_test(1000,100,folder =directory_path ,developer_controller = dc_dic[dc_keys])

            img_directory_path = self.img_save_path + '/'+moduar_name+'_test'+ dc_keys + '_Log'
            img_directory_path_list.append(img_directory_path)
            os.makedirs(img_directory_path, exist_ok=True)
            
        
        n = 0
        for log in directory_path_list:
            log_data_path = log + '/test_log.csv'
            print(log_data_path)
            self.daw_graph('enemy','logs/test_Log/org_test_Log/test_log.csv',log_data_path,lable2=moduar_name, title=moduar_name +' enemy count'+dc_index_list[n],xlable='enemy',img_save_path = img_directory_path_list[n],save_only =save_only)
            self.daw_graph('coin','logs/test_Log/org_test_Log/test_log.csv',log_data_path,lable2=moduar_name, title=moduar_name +' coin count'+dc_index_list[n],xlable='coin',img_save_path = img_directory_path_list[n],save_only =save_only)
            self.daw_graph('step','logs/test_Log/org_test_Log/test_log.csv',log_data_path,lable2=moduar_name, title=moduar_name +' step count'+dc_index_list[n],xlable='step',img_save_path = img_directory_path_list[n],save_only =save_only)
            track_data_path = log+'/trac_log.csv'
            self.darw_track_map(track_data_path,moduar_name +'track map'+dc_index_list[n],img_save_path = img_directory_path_list[n],save_only =save_only)
            
            n +=1 


   
