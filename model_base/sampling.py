import numpy as np
import random

from pymoo.model.sampling import Sampling
from utils import *

class RandomSample(Sampling):
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

class OrderByTimeSample(Sampling):
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
            route_indices = split_at_delimiter(chromosome, return_idx=True)
            for route_index in route_indices:
                if route_index[0] == route_index[1]:
                    continue
                veh_route = chromosome[route_index[0]: route_index[1]]
                customers = self.demand.sort_values('TimeWindow').index
                chromosome[route_index[0]: route_index[1]] = customers[customers.isin(veh_route)]
            X[i] = np.array(chromosome)
            assert sum(X[i] == 0) == (problem.n_vehicles - 1)
        return X