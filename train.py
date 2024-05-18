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

#model = DQN("MultiInputPolicy", env, verbose=1)


env = make_vec_env("MagicTowerEnv-v0",monitor_dir="models")
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    batch_size=1024,
    n_steps=10240 // 1024,  # 等价于 buffer_size / batch_size
    learning_rate=0.0001,
    ent_coef=0.005,  # 类似于beta
    clip_range=0.2,  # 类似于epsilon
    n_epochs=3,
    gamma=0.995,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    vf_coef=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[1024, 1024, 1024, 1024], vf=[1024, 1024, 1024, 1024]),
        lstm_hidden_size=256,  # 内存大小
        n_lstm_layers=1
    ),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
# 添加好奇心机制







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
    
