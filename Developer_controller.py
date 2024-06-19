class Developer_controller:

    def __init__(self,weights) -> None:
        self.weights = weights
        pass

    def add_weight(self,obs):
        new_obs =obs.copy()

        for i in range(len(self.weights)):
            #print(new_obs['module_list'][0][i][1])
            new_obs['module_list'][0][i][1] += self.weights[i]





        return new_obs
        

