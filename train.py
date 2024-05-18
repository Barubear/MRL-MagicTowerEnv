import gymnasium as gym
import MagicTowerEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import sb3_contrib
from sb3_contrib import RecurrentPPO

import torch


#env = gym.make('MagicTowerEnv-v0')

#model = DQN("MultiInputPolicy", env, verbose=1)


vec_env = make_vec_env("MagicTowerEnv-v0",monitor_dir="models")
model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1,
                    learning_rate=0.0001,
                    batch_size=1024,
                    gamma = 0.995)




def train(env,total_timesteps):
    start_msg = evaluate_policy(model,env,n_eval_episodes=20)
    model.learn(total_timesteps, log_interval=4,progress_bar=True,callback = SaceBaseCallback())
    #model.save(model_name)
    del model # remove to demonstrate saving and loading
    model = RecurrentPPO.load('models/best_model')
    trained_msg = evaluate_policy(model,env,n_eval_episodes=20)
    return start_msg,trained_msg



class SaceBaseCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.best = -float('inf')
    
    def _on_step(self) -> bool:
        if self.n_calls%10000 != 0:
            return True
        x , y = ts2xy(load_results("models"),'timesteps')
        mean_reward = sum(y[-100:])/len(y[-100:])
        if mean_reward >self.best:
            self.best = mean_reward
            self.model.save('models/best_model')
        
        return True