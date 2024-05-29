import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import random


class BattleModuleMagicTowerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self,render_mode = "human",size = 10):
        super().__init__()
        self.size = size
        #coin:4
        #enemy:2
        self.origin_map =np.transpose(np.array([
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
         ]))
        self.wall_list = [(1,1),
                          (3,2),(5,2),(8,2),(9,2),
                          (0,3),(1,3),(3,3),(5,3),
                          (8,4),
                          (2,5),(3,5),(5,5),
                          (3,6),(5,6),(6,6),(8,6),(9,6),
                          (1,7),(3,7),(5,7),
                          (7,8),(8,8),
                          (1,9)]

        self.max_step =100000
        self.curr_step = 0
        self.max_HP = 5
        self.curr_HP = self.max_HP
        self.max_enemy_num = 5
        self.curr_nemy_num = self.max_enemy_num
        self.agent_pos ,self.enemy_list  = self.pos_reset()
        self.curr_map = self.origin_map.copy()
        self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 1
        for pos in  self.enemy_list:
            self.curr_map[pos[0],pos[1]] = 2

        self.observation_space = spaces.Box(low=-10, high=10,
                                        shape=(31, 2),dtype=np.int32)
        self.action_space = spaces.Discrete(4)
    
    def pos_reset(self):
        start_pos=(0,0)
        while True:
            new_point = (random.randint(0, 9), random.randint(0, 9))
            if new_point not in self.wall_list:
                start_pos = new_point
                break
        enemy_lsit =[]
        while len(enemy_lsit) < self.max_enemy_num:
            new_point = (random.randint(0, 9), random.randint(0, 9))
            if new_point not in self.wall_list and new_point != start_pos:
                enemy_lsit.append(new_point)
            
        return start_pos,enemy_lsit
    
    def _get_obs(self):
        obs_array = self.wall_list.copy()
        obs_array.append(self.agent_pos)
        obs_array.append((self.curr_HP,self.curr_nemy_num))
        enemy_arry = self.enemy_list.copy()
        obs_array.extend(enemy_arry)
        return np.array(obs_array) 
    
    def _get_info(self,state):
          return{"pos":self.agent_pos,
                 "enemy num": self.curr_nemy_num,
                 "hp":self.curr_HP,
                 "state":state
                }
    def reset(self,seed=None, options=None):
           self.curr_step = 0
           self.curr_HP = self.max_HP          
           self.agent_pos ,self.enemy_list  = self.pos_reset()
           self.curr_map = self.origin_map.copy()
           self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 1
           for pos in  self.enemy_list:
               self.curr_map[pos[0],pos[1]] = 2
           self.curr_nemy_num = self.max_enemy_num
           

           return self._get_obs() , self._get_info(False)
    def step(self, action):
          next_x= self.agent_pos[0]
          next_y =self.agent_pos[1]
          reward = 0
          terminated = False
          truncated =False
          ifdone =False
          self.curr_step +=1
          if self.curr_step>= self.max_step:
              truncated = True
              return self._get_obs(), reward, terminated, truncated, self._get_info()
          
          if(action == 0):#up
              next_y-=1
          elif(action == 1):#down
              next_y+=1
          elif(action == 2):#right
              next_x+=1
          elif(action == 3):#left
              next_x-=1
          
          
          if(next_x < 0 or next_x >=7 or next_y < 0 or next_y >=7):
              reward -=1
          else:
              # wall:-1
              if(self.curr_map[next_x,next_y] == -1):
                  reward -=1
                  #self.curr_visit_map[self.agent_pos[0],self.agent_pos[1]] +=1
              # way
              if(self.curr_map[next_x,next_y] == 0):
                  reward -=0.1 
                  
                  self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                  self.curr_map[next_x,next_y] = 1
                  self.agent_pos = [next_x,next_y]
            
              #enemy
              if(self.curr_map[next_x,next_y] == 2):
                    ifdone = True
                    if random.random() < 0.4:  # 50% chance of winning
                        reward +=200
                    else:  #
                        reward += 100
                        self.curr_HP -= 1
                    
                    self.curr_nemy_num-=1
                    if self.curr_HP <= 0:
                        reward -= 500
                        terminated = True
                    if self.curr_nemy_num == 0:
                        reward +=500
                        terminated = True
                    self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                    self.curr_map[next_x,next_y] = 1
                    self.agent_pos = [next_x,next_y]
                    for enemy in self.enemy_list:
                        if enemy[0] == next_x and enemy[1] == next_y:
                            enemy = (-1,-1)
            #enemy
              if(self.curr_map[next_x,next_y] == 3):
                  if self.curr_HP >1:
                      reward -= self.curr_HP*100
                  else:
                   reward += 1000
          
          observation = self._get_obs()
          info = self._get_info(ifdone)

          
          
          
          return observation, reward, terminated, truncated, info
    


register(
    id='BattleModuleMagicTowerEnv',
    entry_point='Envs.modularEnv.BattleModuleMagicTowerEnv:BattleModuleMagicTowerEnv',
    max_episode_steps=50000,
)