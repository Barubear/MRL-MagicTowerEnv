
import gymnasium as gym
import MagicTowerEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

import torch


#env = gym.make('MagicTowerEnv-v0')

#model = DQN("MultiInputPolicy", env, verbose=1)


vec_env = make_vec_env("MagicTowerEnv-v0",monitor_dir="models")
model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1)



def train(model,env,total_timesteps,model_name ="PPO_cartpole" ):
    start_msg = evaluate_policy(model,env,n_eval_episodes=100)
    model.learn(total_timesteps, log_interval=4,progress_bar=True,callback = SaceBaseCallback())
    #model.save(model_name)
    del model # remove to demonstrate saving and loading
    model = RecurrentPPO.load('models/best_model')
    trained_msg = evaluate_policy(model,env,n_eval_episodes=100)
    return start_msg,trained_msg

def test(model,env,max_step = 100,print_log_step = 10):
    obs = env.reset()
    over =False
    step =0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while True:
        
        action, _states = model.predict(obs)
        obs_tensor = torch.tensor(obs).to(device)
        _states_tensor = torch.tensor(_states,dtype=torch.float32).to(device)
        episode_starts = torch.tensor([True] ,dtype=torch.float32).to(device)
        state_value = model.policy.predict_values(obs_tensor,_states_tensor ,episode_starts)

        obs, rewards, dones, info  = env.step(action)
        
        
        

        if step % print_log_step == 0:
            print(state_value)
            print(info)
        if dones or step >=max_step:
            info = env.reset()
            break
        step +=1

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

#print(train(model,vec_env,5000000))
del model
model = RecurrentPPO.load("models/best_model" )
test(model,vec_env,200,20) 
# 假设model是你训练好的模型对象

