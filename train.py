
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from Data_processor import Data_Processor 
import numpy as np


def train(model,env,total_timesteps, save_path,log_path,test_times,testonly =False):

    if not testonly:
        start_msg = evaluate_policy(model,env,n_eval_episodes=test_times)
        model.learn(total_timesteps, log_interval=4,progress_bar=True,callback = SaceBaseCallback(save_path,log_path))
        
        del model # remove to demonstrate saving and loading
        model = RecurrentPPO.load(save_path)
        trained_msg = evaluate_policy(model,env,n_eval_episodes=test_times)
        print(start_msg ,trained_msg)

    print('model test:')

    state_value_list = []
    model = RecurrentPPO.load(save_path)
    dp = Data_Processor(env,model)
    for i in range(test_times):
        step = 0
        obs= env.reset()
        while True:

                action, _states = model.predict(obs)
                obs, rewards, dones, info  = env.step(action)
                state_value_list.append(dp.get_state_value(_states,obs))
                step +=1
                if dones or step >= 100:
                    print(step,info)
                    break
    
    state_value_list = [value.detach().cpu().numpy() for value in state_value_list]
    state_value_list = np.array(state_value_list)

    print('state_value:')
    print('max:' + str(state_value_list.max()))
    print('mean:' + str(state_value_list.mean()))
    print('min:' + str(state_value_list.min()))

class SaceBaseCallback(BaseCallback):
    def __init__(self, save_path,log_path):
        super().__init__(verbose=0)
        self.best = -float('inf')
        self.save_path = save_path
        self.log_path = log_path
        self.best_step = 0
    def _on_step(self) -> bool:
        
        if self.n_calls%1000 != 0:
            
            return True
        x , y = ts2xy(load_results(self.log_path),'timesteps')
        mean_reward = sum(y[-100:])/len(y[-100:])
        print(self.best_step)
        if mean_reward >self.best:
            self.best = mean_reward
            self.best_step = self.n_calls
            print(self.n_calls,self.best)
            self.model.save(self.save_path)
        
        return True
    
