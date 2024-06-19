class Developer_controller:

    def __init__(self,weights) -> None:
        self.weights = weights
        pass

    def add_weight(self,obs):
        new_obs =obs.copy()

        for i in range(len(self.weights)):

            new_obs['map'][i][1] += self.weights[i]





        return new_obs
        

