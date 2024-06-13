from sb3_contrib import RecurrentPPO
import gymnasium as gym
from abc import ABC, abstractmethod
import numpy as np
import torch


class modulars:

    def __init__(self,path,env) :
        self.module = RecurrentPPO.load(path)
        self.device = torch.device('cuda' )
        self.main_env =env

    @abstractmethod
    def get_obs(self):
        pass


    def get_state_value(self,env):
        obs = self.get_obs()
        action, _states = self.model.predict(obs)
        obs_tensor_dict = {key: torch.as_tensor(obs, device=self.device) for (key, obs) in obs.items()}


        _states_tensor = torch.tensor(_states,dtype=torch.float32).to(self.device)
        episode_starts = torch.tensor([True] ,dtype=torch.float32).to(self.device)
        state_value = self.model.policy.predict_values(obs_tensor_dict,_states_tensor ,episode_starts)

        return action,state_value 

    @abstractmethod
    def state_updata(self,x_pos= 0,y_pos = 0):
        pass

    @abstractmethod
    def reset(self):
        pass


class BattleModular(modulars):
    def __init__(self,path,env):
        super().__init__(path,env)
        
        self.origin_enemy_list = np.array([
                (0,3),(2,1),(2,3),
            ], dtype=int)
        self.enemy_list= self.origin_enemy_list.copy()
        
        
    def get_obs(self):
        return {
                "map":np.array(self.env.curr_map, dtype=int),
                "agent": np.array(self.env.agent_pos, dtype=int),
                "hp/cur_enemy":np.array((self.curr_HP,self.curr_nemy_num), dtype=int),
                "target": np.array(self.enemy_list, dtype=int),
        }
    

    def state_updata(self,x_pos= 0,y_pos = 0):
        self.enemy_list = [(-1, -1) if (e[0] == x_pos and e[1] == y_pos) else e for e in self.enemy_list]
        

    def reset(self):
        self.enemy_list= self.origin_enemy_list.copy()

class CoinModular(modulars):
    def __init__(self,path,env) :
        super.__init__(path,env)
        self.origin_coin_list = np.array([
            (4,0),(5,2),(3,5),(5,5)
        ], dtype=int)
        
        self.coin_list= self.origin_coin_list.copy()
    

    def get_obs(self):
            return {
                "map":np.array(self.env.curr_map, dtype=int),
                "agent": np.array(self.env.agent_pos, dtype=int),
                "target": np.array(self.coin_list, dtype=int),
            }
    
    
    def state_updata(self,x_pos= 0,y_pos = 0):
        self.coin_list = [(-1, -1) if (e[0] == x_pos and e[1] == y_pos) else e for e in self.coin_list]
        
    def reset(self):
        self.coin_list= self.origin_coin_list.copy()


class KeyModular(modulars):
    def __init__(self,path,env) :
        super.__init__(path,env)
        self.key_pos  = (0,2)
        
        
    

    def get_obs(self):
            return {
                "map":np.array(self.env.curr_map, dtype=int),
                "agent": np.array(self.env.agent_pos, dtype=int),
                "target": np.array(self.key_pos, dtype=int),
            }
    
    
    def state_updata(self,x_pos= 0,y_pos = 0):
        self.key_pos  = (-1,-1)
        
    def reset(self):
        self.key_pos =(0,2)