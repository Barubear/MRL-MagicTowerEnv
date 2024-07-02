import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import random


class KeyModuleMagicTowerEnv_6x6(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self,render_mode = "human",size = 6):
        super().__init__()
        self.size = size
        #coin:4
        #enemy:2
        self.origin_map =np.transpose(np.array([
         [ 3, 0, 0, 0, 0, 0],
         [ 0,-1, 0, 0,-1, 0],
         [ 5, 0, 0, 0, 0, 0],
         [ 0,-1, 0,-1, 0,-1],
         [ 0,-1, 0,-1, 0,-1],
         [ 0, 0, 0, 0, 0, 0],
         
         ], dtype=int))
        
        
        
        
        self.key_pos  = (0,2)
        self.have_key = False
        self.startPos_index=0
        self.startPos=[(1,5),(2,0),(5,5),(5,0)]
        self.agent_pos = self.startPos[ self.startPos_index % len(self.startPos)]
        self.curr_map = self.origin_map.copy()
        self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 1
        

        self.observation_space = spaces.Dict(
            {
                "map":spaces.Box(-10, 10, shape=(size,size), dtype=int),
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(-1, size - 1, shape=(2,), dtype=int),

            }
        )



        self.action_space = spaces.Discrete(4)
    
    def pos_reset(self):
        start_pos=(0,0)
        while True:
            new_point = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if self.origin_map[new_point[0],new_point[1]] == 0  :
                start_pos = new_point
                break
            
        return start_pos
    
    def _get_obs(self):
        
        return {
                "map":np.array(self.curr_map, dtype=int),
                "agent": np.array(self.agent_pos, dtype=int),
                "target": np.array(self.key_pos, dtype=int),
            }
    
    def _get_info(self,state):
          return{"pos":self.agent_pos,
                 "key":self.have_key,
                 "state":state
                }
    def reset(self,seed=None, options=None):
           
           self.key_pos  = (0,2)
           self.curr_map = self.origin_map.copy()          
           self.startPos_index += 1
           self.agent_pos = self.startPos[ self.startPos_index % len(self.startPos)]
           self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 1
           self.have_key = False
           

           return self._get_obs() , self._get_info(False)
    
    def step(self, action):
          next_x= self.agent_pos[0]
          next_y =self.agent_pos[1]
          reward = 0
          terminated = False
          truncated =False
          ifdone =False
          if(action == 4):#quit
                if self.curr_HP >1:
                      reward -= self.curr_HP*500
                else:
                   reward += 100
                terminated = True
          else:
            if(action == 0):#up
              next_y-=1
            elif(action == 1):#down
              next_y+=1
            elif(action == 2):#right
              next_x+=1
            elif(action == 3):#left
              next_x-=1
          
          
            if(next_x < 0 or next_x >=self.size or next_y < 0 or next_y >=self.size):
              reward -=1
            else:
              # wall:-1
              if(self.curr_map[next_x,next_y] == -1):
                  reward -=2
              # way
              if(self.curr_map[next_x,next_y] == 0):
                  reward -=1
                  self._update_agent_position(next_x,next_y)
            
              #key
              if(self.curr_map[next_x,next_y] == 5):
                    ifdone = True
                    self.have_key = True
                    
                    reward +=10
                    
                    self._update_agent_position(next_x,next_y)

                    self.key_pos = (-1,-1)
              
              if(self.curr_map[next_x,next_y] == 3):
                  
                if self.have_key == True:
                      reward+=20
                      ifdone = True
                      terminated = True
                      self._update_agent_position(next_x,next_y)
                else:
                      reward-=20

                
              
          
            

          
          
          observation = self._get_obs()
          info = self._get_info(ifdone)
          return observation, reward, terminated, False, info
    def _update_agent_position(self, next_x,next_y):
        self.curr_map[self.agent_pos[0], self.agent_pos[1]] = 0
        self.curr_map[next_x, next_y] = 1
        self.agent_pos = (next_x, next_y)


register(
    id='KeyModuleMagicTowerEnv_6x6',
    entry_point='Envs.modularEnv.KeyModuleMagicTowerEnv_6x6:KeyModuleMagicTowerEnv_6x6',
    max_episode_steps=1000,
)