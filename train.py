
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results,ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

def train(model,env,total_timesteps, save_path,log_path):
    start_msg = evaluate_policy(model,env,n_eval_episodes=20)
    model.learn(total_timesteps, log_interval=4,progress_bar=True,callback = SaceBaseCallback(save_path,log_path))
    #model.save(model_name)
    del model # remove to demonstrate saving and loading
    model = RecurrentPPO.load(save_path)
    trained_msg = evaluate_policy(model,env,n_eval_episodes=20)
    return start_msg,trained_msg



class SaceBaseCallback(BaseCallback):
    def __init__(self, save_path,log_path,verbose: int = 0):
        super().__init__(verbose)
        self.best = -float('inf')
        self.save_path = save_path
        self.log_path = log_path
    
    def _on_step(self) -> bool:
        if self.n_calls%10000 != 0:
            return True
        x , y = ts2xy(load_results(self.log_path),'timesteps')
        mean_reward = sum(y[-100:])/len(y[-100:])
        if mean_reward >self.best:
            self.best = mean_reward
            self.model.save(self.save_path)
        
        return True
    
