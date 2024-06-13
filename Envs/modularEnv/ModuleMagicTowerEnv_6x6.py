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
         [ 3, 0, 0, 0, 0, 0],
         [ 0,-1, 2, 0,-1, 0],
         [ 5, 0, 0, 0, 0, 0],
         [ 2,-1, 2,-1, 0,-1],
         [ 0,-1, 0,-1, 0,-1],
         [ 1, 0, 0, 4, 0, 4],
         
         ], dtype=int))
        
        self.start_pos = (0,5)
        self.agent_pos = self.start_pos.copy()
        self.curr_map = self.origin_map.copy()

        battble_modular = BattleModular('trained_modules/BattleModule/Battle_minusReward_for_lose',self)
        self.max_enemy_num = len(battble_modular.origin_enemy_list)
        self.curr_nemy_num = self.max_enemy_num
        self.max_HP = self.max_enemy_num-1
        self.curr_HP = self.max_HP
        
        
        coin_modular = CoinModular('trained_modules\CoinModule\Coin_best_mode',self)

        key_modular = KeyModular('trained_modules\KeyModule\Key_best_mode',self)

        self.curr_modular_index = 2 # defaule == key_modular
        self.modular_action_list =[(0,0),(0,0),(0,0)]
        self.modualr_list =[battble_modular,coin_modular,key_modular]
        self.observation_space = spaces.Dict(
            {
                "map":spaces.Box(-10, 10, shape=(size,size), dtype=int),
                "module_list": spaces.Box(0, 50, shape=(3,2), dtype=float),
                "curr_module":spaces.Box(0, 5, shape=(2,), dtype=int),
            
            }
        )
        self.action_space = spaces.Discrete(3)
    
    
    
    def _get_obs(self):
        return{

                "map":np.array(self.curr_map, dtype=int),
                "module_list": np.array(self.modular_action_list, dtype=int),
                "curr_module":np.array(self.modular_action_list[self.curr_modular_index], dtype=int),

        }
        pass
    

    def _get_info(self,state):
        pass
    

    def reset(self,seed=None, options=None):
           
        pass
    
    def agent_step(self, action):
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