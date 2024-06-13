import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import random
from sb3_contrib import RecurrentPPO
import torch
from modulars import BattleModular,CoinModular,KeyModular

class ModuleMagicTowerEnv_6x6(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self,render_mode = "human",size = 6):
        super().__init__()
        self.size = size
        #coin:4
        #enemy:2
        self.origin_map =np.transpose(np.array([
         [ 3, 0, 0, 0, 4, 0],
         [ 0,-1, 2, 0,-1, 0],
         [ 5, 0, 0, 0, 0, 4],
         [ 2,-1, 2,-1, 0,-1],
         [ 0,-1, 0,-1, 0,-1],
         [ 1, 0, 0, 4, 0, 4],
         
         ], dtype=int))
        
        self.start_pos = (0,5)
        self.agent_pos = self.start_pos
        self.curr_map = self.origin_map.copy()

        self.battble_modular = BattleModular('modules\BattleModule\Battle_minusReward_for_lose',self)
        self.max_enemy_num = len(self.battble_modular.origin_enemy_list)
        self.curr_nemy_num = self.max_enemy_num
        self.max_HP = self.max_enemy_num-1
        self.curr_HP = self.max_HP
        
        
        self.coin_modular = CoinModular('trained_modules\CoinModule\Coin_best_mode',self)
        self.max_coin_num = 4
        self.curr_coin_num = self.max_coin_num


        self.key_modular = KeyModular('trained_modules\KeyModule\Key_best_mode',self)
        self.have_key = False

        self.curr_modular_index = 2 # defaule == key_modular
        self.modular_predict_list =[(0,0),(0,0),(0,0)]
        self.modular_action_list =[0,0,0]
        self.modualr_list =[self.battble_modular, self.coin_modular, self.key_modular]
        self.observation_space = spaces.Dict(
            {
                "map":spaces.Box(-10, 10, shape=(size,size), dtype=int),
                "module_list": spaces.Box(0, 50, shape=(len(self.modular_predict_list),2), dtype=float),
                "curr_module":spaces.Box(0, 5, shape=(2,), dtype=float),
            
            }
        )
        self.action_space = spaces.Discrete(3)
    
    
    
    def _get_obs(self):
        return{

                "map":np.array(self.curr_map, dtype=int),
                "module_list": np.array(self.modular_predict_list, dtype=float),
                "curr_module":np.array(self.modular_predict_list[self.curr_modular_index], dtype=float),

        }
        
    

    def _get_info(self):
        return{
            "hp/enemy":(self.curr_HP,self.curr_nemy_num),
            "coin":self.curr_coin_num,


        }
        
    

    def reset(self,seed=None, options=None):
        self.agent_pos = self.start_pos
        self.curr_map = self.origin_map.copy()
        self.curr_nemy_num = self.max_enemy_num
        self.curr_HP = self.max_HP 
        self.have_key = False
        self.curr_coin_num = self.max_coin_num

        for m in self.modualr_list:
            m.reset()

        return self._get_obs() , self._get_info()
        
    
    def step(self, action):
        self.curr_modular_index =action
        modular_action = self.modular_action_list[action]
        next_x= self.agent_pos[0]
        next_y =self.agent_pos[1]
        reward = 0
        terminated = False
        truncated =False


        if(modular_action == 0):#up
            next_y-=1
        elif(modular_action == 1):#down
            next_y+=1
        elif(modular_action == 2):#right
            next_x+=1
        elif(modular_action == 3):#left
            next_x-=1

        
        if(next_x < 0 or next_x >=self.size or next_y < 0 or next_y >=self.size):
              reward -=1
        else:
            # wall:-1
            if(self.curr_map[next_x,next_y] == -1):
                reward -=1
            # way
            elif(self.curr_map[next_x,next_y] == 0):
                reward -=0.3
                self._update_agent_position(next_x,next_y)
            #enemy
            elif(self.curr_map[next_x,next_y] == 2):
                if random.random() < 0.5:  # 50% chance of winning
                    reward +=5
                else:  #
                    reward -= 2
                    self.curr_HP -= 1
                
                self.curr_nemy_num-=1

                if self.curr_HP <= 0:
                    reward -= 10
                    terminated = True
                    
                self._update_agent_position(next_x,next_y)
                self.battble_modular.state_updata(next_x,next_y)
            #exit
            elif(self.curr_map[next_x,next_y] == 3):  
                if self.have_key == True:
                      reward+=10
                      terminated = True
                      self._update_agent_position(next_x,next_y)
                else:
                      reward-=10
            #coin
            elif(self.curr_map[next_x,next_y] == 4):  
                reward +=5
                self.curr_coin_num -= 1    
                    
                self._update_agent_position(next_x,next_y)
                self.coin_modular.state_updata(next_x,next_y)



            #key
            elif(self.curr_map[next_x,next_y] == 5):
                self.have_key = True
                reward +=10  
                self._update_agent_position(next_x,next_y) 
                self.key_modular.state_updata() 


        
        for i in range(len(self.modualr_list)) :
            modular = self.modualr_list[i]
            new_action, state_value =modular.get_state_value(modular.get_obs())
            self.modular_predict_list[i] = (new_action, state_value)
            self.modular_action_list[i] =new_action
        


        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
        








    def _update_agent_position(self, next_x,next_y):
        self.curr_map[self.agent_pos[0], self.agent_pos[1]] = 0
        self.curr_map[next_x, next_y] = 1
        self.agent_pos = (next_x, next_y)


register(
    id='ModuleMagicTowerEnv_6x6',
    entry_point='Envs.modularEnv.ModuleMagicTowerEnv_6x6:ModuleMagicTowerEnv_6x6',
    max_episode_steps=1000,
)