import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import random
from sb3_contrib import RecurrentPPO
import torch


class ModuleMagicTowerEnv_6x6(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self,render_mode = "human",size = 6):
        super().__init__()
        self.size = size
        #coin:4
        #enemy:2
        self.origin_map =np.transpose(np.array([
         [ 3, 0, 0, 0, 0, 0],
         [ 0,-1, 2, 0,-1, 0],
         [ 5, 0, 0, 0, 0, 0],
         [ 2,-1, 2,-1, 0,-1],
         [ 0,-1, 0,-1, 0,-1],
         [ 1, 0, 0, 4, 0, 4],
         
         ], dtype=int))
        

        #BattleModule
        self.BattleModule = RecurrentPPO.load('modules/BattleModule/Battle_best_model882000')
        
        self.origin_enemy_list = np.array([
            (0,3),(2,1),(2,3),
        ], dtype=int)
        self.enemy_list= self.origin_enemy_list.copy()
        self.max_enemy_num = len(self.origin_enemy_list)
        self.curr_nemy_num = self.max_enemy_num
        self.max_HP = self.max_enemy_num-1
        self.curr_HP = self.max_HP


        #CoinModule
        self.CoinModule = RecurrentPPO.load('modules/CoinModule/Coin_best_model')

        self.origin_coin_list = np.array([
            (4,0),(5,2),(3,5),(5,5)
        ], dtype=int)
        
        self.coin_list= self.origin_coin_list.copy()
        self.max_coin_num = len(self.origin_coin_list)
        self.curr_coin_num = self.max_coin_num


        #keyModule
        self.KeyModule = RecurrentPPO.load('modules/KeyModule/Key_best_mode')
        self.key_pos  = (0,2)
        self.have_key = False


        
        #controller
        self.device = torch.device('cuda' )
        self.module_action_list = [0, 0, 0]
        self.agent_pos = (0,5)
        self.curr_module_index = 2 # default == key module
        self.curr_module_state = False # False == runing
        self.curr_map = self.origin_map.copy()
        self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 1
        
        

        self.observation_space = spaces.Dict(
            {
                "map":spaces.Box(-10, 10, shape=(size,size), dtype=int),
                "module_list": spaces.Box(0, 50, shape=(1,3), dtype=float),
                "curr_module":spaces.Box(0, 5, shape=(2,), dtype=int),
            
            }
        )



        self.action_space = spaces.Discrete(3)
    
    
    
    def _get_main_obs(self):
        
        return {
                "map":np.array(self.curr_map, dtype=int),
                "module_list":np.array(self.module_action_list, dtype=float),
                "curr_module":np.array((self.curr_module_index, self.curr_module_state), dtype=int,)
            }
    

    def _get_battle_obs(self):
        
        return {
                "map":np.array(self.curr_map, dtype=int),
                "agent": np.array(self.agent_pos, dtype=int),
                "hp/cur_enemy":np.array((self.curr_HP,self.curr_nemy_num), dtype=int),
                "target": np.array(self.enemy_list, dtype=int),
            }

    def _get_coin_obs(self):
        
        return {
                "map":np.array(self.curr_map, dtype=int),
                "agent": np.array(self.agent_pos, dtype=int),
                "target": np.array(self.coin_list, dtype=int),
            }

    def _get_key_obs(self):
        
        return {
                "map":np.array(self.curr_map, dtype=int),
                "agent": np.array(self.agent_pos, dtype=int),
                "target": np.array(self.key_pos, dtype=int),
            }


    def _get_info(self,state):
          return{"pos":self.agent_pos,
                 "enemy num": self.curr_nemy_num,
                 "hp":self.curr_HP,
                 "state":state
                }
    

    def reset(self,seed=None, options=None):
           
           self.enemy_list= self.origin_enemy_list.copy()
           self.curr_map = self.origin_map.copy()          
           self.startPos_index += 1
           self.agent_pos = self.startPos[ self.startPos_index % len(self.startPos)]
           self.curr_map[self.agent_pos[0],self.agent_pos[1]] = 1
           self.curr_nemy_num = self.max_enemy_num
           self.curr_HP = self.max_HP
           

           return self._get_obs() , self._get_info(False)
    
    def agent_step(self, action):
          
          
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
                  reward -=1
              # way
              if(self.curr_map[next_x,next_y] == 0):
                  reward -=0.3
                  self._update_agent_position(next_x,next_y)
            
              #enemy
              if(self.curr_map[next_x,next_y] == 2):
                    ifdone = True

                    if random.random() < 0.5:  # 40% chance of winning
                        reward +=5
                    else:  #
                        reward -= 2
                        self.curr_HP -= 1
                    
                    self.curr_nemy_num-=1

                    if self.curr_HP <= 0:
                        reward -= 10
                        terminated = True

                    elif self.curr_nemy_num == 0:
                        reward +=10
                        terminated = True
                    
                    self._update_agent_position(next_x,next_y)

                    self.enemy_list = [(-1, -1) if (e[0] == next_x and e[1] == next_y) else e for e in self.enemy_list]
                            #coin
              #coin
              if(self.curr_map[next_x,next_y] == 4):
                    ifdone = True
                    reward +=5
                    self._update_agent_position(next_x,next_y)
                    self.curr_coin_num-=1
                    self.coin_list = [(-1, -1) if (e[0] == next_x and e[1] == next_y) else e for e in self.coin_list]
                    ifdone = True
                    if self.curr_coin_num == 0:
                        reward+=10
                        
                        terminated = True
              #key
              if(self.curr_map[next_x,next_y] == 5):
                    ifdone = True
                    self.have_key = True
                    reward +=10                   
                    self._update_agent_position(next_x,next_y)
                    self.key_pos = (-1,-1)
              #exit
              if(self.curr_map[next_x,next_y] == 3):
                  
                if self.have_key == True:
                      reward+=10
                      ifdone = True
                      terminated = True
                      self._update_agent_position(next_x,next_y)
                else:
                      reward-=10

                ifdone = True
                terminated = True
              
          
            

          
          
          observation = self._get_main_obs()
          info = self._get_info(ifdone)
          return observation, reward, terminated, False, info
    
    def controller_step(self,model):
        
        
        pass



    def _update_agent_position(self, next_x,next_y):
        self.curr_map[self.agent_pos[0], self.agent_pos[1]] = 0
        self.curr_map[next_x, next_y] = 1
        self.agent_pos = (next_x, next_y)


register(
    id='ModuleMagicTowerEnv_6x6',
    entry_point='Envs.modularEnv.ModuleMagicTowerEnv_6x6:ModuleMagicTowerEnv_6x6',
    max_episode_steps=1000,
)