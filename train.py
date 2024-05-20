import gymnasium as gym
from gymnasium import spaces
import MagicTowerEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib.common import utils


from sb3_contrib import RecurrentPPO

import torch


#env = gym.make('MagicTowerEnv-v0')

#model = DQN("MultiInputLstmPolicy", env, verbose=1)












def train(model,env,total_timesteps,path):
    start_msg = evaluate_policy(model,env,n_eval_episodes=20)
    model.learn(total_timesteps, log_interval=4,progress_bar=True,callback = SaceBaseCallback(path))
    #model.save(model_name)
    del model # remove to demonstrate saving and loading
    model = RecurrentPPO.load(path)
    trained_msg = evaluate_policy(model,env,n_eval_episodes=20)
    return start_msg,trained_msg



class SaceBaseCallback(BaseCallback):
    def __init__(self, path,verbose: int = 0):
        super().__init__(verbose)
        self.best = -float('inf')
        self.path = path
    
    def _on_step(self) -> bool:
        if self.n_calls%10000 != 0:
            return True
        x , y = ts2xy(load_results("models"),'timesteps')
        mean_reward = sum(y[-100:])/len(y[-100:])
        if mean_reward >self.best:
            self.best = mean_reward
            self.model.save(self.path)
        
        return True
    
