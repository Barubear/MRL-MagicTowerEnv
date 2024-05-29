import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3.common.results_plotter import load_results,ts2xy


def pos_reset(max_enemy_num):
        start_pos=(0,0)
        while True:
            new_point = (random.randint(0, 9), random.randint(0, 9))
            if origin_map[new_point[0],new_point[1]] != -1 :
                start_pos = new_point
                break
        enemy_lsit =[]
        while len(enemy_lsit) < max_enemy_num:
            new_point = (random.randint(0, 9), random.randint(0, 9))
            if  origin_map[new_point[0],new_point[1]] != -1  and new_point != start_pos:
                enemy_lsit.append(new_point)
            
        return start_pos,enemy_lsit







size = 10
        #coin:4
        #enemy:2
origin_map =np.transpose(np.array([
[ 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
[ 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0,-1, 0, 0,-1,-1],
         [-1,-1, 0,-1, 0,-1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
         [ 0, 0,-1,-1, 0,-1, 0, 0, 0, 0],
         [ 0, 0, 0,-1, 0,-1,-1, 0,-1,-1],
         [ 0,-1, 0,-1, 0,-1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0,-1,-1, 0],
         [ 0,-1, 0, 0, 1, 0, 0, 0, 0, 0],
], dtype=int))
        
max_step =100000
curr_step = 0
max_HP = 5
curr_HP = max_HP
max_enemy_num = 5
curr_nemy_num =max_enemy_num
agent_pos ,enemy_list  = pos_reset(max_enemy_num)
curr_map = origin_map.copy()
curr_map[agent_pos[0],agent_pos[1]] = 1
for pos in  enemy_list:
        curr_map[pos[0],pos[1]] = 2

observation_space = spaces.Dict(
            {
                
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "hp":spaces.Box(0, max_HP, shape=(), dtype=int),
                "map":spaces.Box(-10, 10, shape=(size,size), dtype=int),
                "target": spaces.Box(-1, size - 1, shape=(5,2), dtype=int),

            }
        )
#print(observation_space)

obs ={
                
                "agent": np.array(agent_pos, dtype=int),
                "hp":np.array(curr_HP,  dtype=int),
                "map":np.array(curr_map,  dtype=int),
                "target": np.array(enemy_list,  dtype=int),
}
print(obs in observation_space)

