import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import random


class Celx100MagicTowerEnv_with_winRate(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode = "human",size = 7):
        super().__init__()
        self.size = size
        #coin:4
        #enemy:2
        self.origin_map =np.transpose(np.array([
         [ 0, 2, 0, 0, 3, 0, 0, 0, 0, 0],
         [ 4,-1, 0, 0, 2, 0, 0, 0, 0, 4],
         [ 0, 0, 0,-1, 0,-1, 0, 0,-1,-1],
         [-1,-1, 0,-1, 0,-1, 0, 0, 0, 2],
         [ 0, 0, 0, 0, 5, 0, 0, 0,-1, 0],
         [ 0, 0,-1,-1, 2,-1, 0, 0, 0, 0],
         [ 0, 0, 2,-1, 0,-1,-1, 0,-1,-1],
         [ 0,-1, 0,-1, 0,-1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0,-1,-1, 0],
         [ 4,-1, 0, 0, 1, 0, 0, 0, 0, 0],
         ]))
        
        self.max_step =100000
        self.curr_step = 0
      
        self.start_pos = [4,9]
        self.agent_pos = self.start_pos.copy()

        self.max_HP = 5
        self.curr_HP = self.max_HP

        self.max_coin_num = 4
        self.curr_coin_num = self.max_coin_num

        self.max_enemy_num = 5
        self.curr_nemy_num = self.max_enemy_num
        self.if_have_key =False
        
        self.key_index = 0
        
        #self.curr_visit_map = self.origin_visit_map.copy()
        self.observation_space = spaces.Box(low=-20, high=20,
                                        shape=(size, size),dtype=np.int32)
        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        obs_array = self.curr_map.copy()
        agent_value = 10 + self.curr_HP
        if self.if_have_key!= True:
            agent_value = -agent_value
        obs_array[self.agent_pos[0],self.agent_pos[1]] = agent_value
        obs_array = obs_array.astype(np.int32)
        return obs_array
    
    def _get_info(self):
          return{"pos":self.agent_pos,
                 "if_have_key":self.if_have_key,
                 "enemy num": self.curr_nemy_num,
                 "hp":self.curr_HP,
                 "coin num":self.curr_coin_num }
    def reset(self,seed=None, options=None):
           self.curr_step = 0
           self.if_have_key =False
           self.curr_HP = self.max_HP          
           self.curr_map = self.origin_map.copy()
           self.agent_pos = self.start_pos.copy()
           self.curr_coin_num = self.max_coin_num
           self.curr_nemy_num = self.max_enemy_num
           

           return self._get_obs() , self._get_info()
    def step(self, action):
          next_x= self.agent_pos[0]
          next_y =self.agent_pos[1]
          reward = 0
          terminated = False
          truncated =False
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
              reward -=10
          else:
              # wall:-1
              if(self.curr_map[next_x,next_y] == -1):
                  reward -=10
                  #self.curr_visit_map[self.agent_pos[0],self.agent_pos[1]] +=1
              # way
              if(self.curr_map[next_x,next_y] == 0):
                  reward -=0.5  
                  
                  self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                  self.curr_map[next_x,next_y] = 1
                  self.agent_pos = [next_x,next_y]
                  
              # coin
              if(self.curr_map[next_x,next_y] == 4):
                  reward +=100
                  self.curr_coin_num-=1
                  self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                  self.curr_map[next_x,next_y] = 1
                  self.agent_pos = [next_x,next_y]
              #key
              if(self.curr_map[next_x,next_y] == 5):
                  reward +=500
                  
                  self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                  self.curr_map[next_x,next_y] = 1
                  self.agent_pos = [next_x,next_y]
                  self.if_have_key = True
              #exit
              if(self.curr_map[next_x,next_y] == 3):
                  if(self.if_have_key):
                    reward +=1000
                    terminated = True
                    self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                    self.curr_map[next_x,next_y] = 1
                    self.agent_pos = [next_x,next_y]
                  else:
                    reward -=100
              #enemy
              if(self.curr_map[next_x,next_y] == 2):
                    if random.random() < 0.3:  # 50% chance of winning
                        reward +=50*self.curr_HP
                    else:  #
                        reward -= 20*self.curr_HP
                       
                    self.curr_HP -= 1
                    self.curr_nemy_num-=1
                    if self.curr_HP <= 0:
                        reward -= 1000
                        terminated = True
                     
                    self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 0
                    self.curr_map[next_x,next_y] = 1
                    self.agent_pos = [next_x,next_y]
                    
          
          observation = self._get_obs()
          info = self._get_info()

          
          
          
          return observation, reward, terminated, truncated, info
    


register(
    id='Celx100MagicTowerEnv_with_winRate',
    entry_point='Celx100MagicTowerEnv_with_winRate:Celx100MagicTowerEnv_with_winRate',
    max_episode_steps=100000,
)