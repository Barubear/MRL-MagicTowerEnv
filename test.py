import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3.common.results_plotter import load_results,ts2xy
from sb3_contrib import RecurrentPPO

from modulars import BattleModular,CoinModular,KeyModular

class test_env:
        def __init__(self) -> None:

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


modular_predict_list =[(0,0),(0,0),(0,0)]
modular_action_list =[0,0,0]

a = np.array(modular_predict_list, dtype=float)
b = np.array(modular_predict_list[0], dtype=float)
print(a.shape)
print(b.shape)