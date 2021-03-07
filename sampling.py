import numpy as np
import random

from pymoo.model.sampling import Sampling

class MySampling(Sampling):
    def __init__(self, demand, n_vehicles):
        super().__init__()
        self.demand = demand
        self.n_vehicles = n_vehicles

    def _do(self, problem, n_samples, **kwargs):
        chromosome_len = len(self.demand) + self.n_vehicles - 1
        X = np.full((n_samples, chromosome_len), 0, dtype=np.int)
        for i in range(n_samples):
            chromosome = list(self.demand.index) + [0] * (self.n_vehicles - 1)
            random.shuffle(chromosome)
            X[i] = np.array(chromosome)
            assert sum(X[i] == 0) == (problem.n_vehicles - 1)
        return X