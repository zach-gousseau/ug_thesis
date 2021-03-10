import copy
import numpy as np
import random
from utils import *
from collections import Counter

from pymoo.model.crossover import Crossover


class CrossRoutes(Crossover):
    def __init__(self, data):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.data = data

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=np.object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            a, b = X[0, k], X[1, k]

            co_point = random.randint(0, self.data['n_vehicles'] - 2)
            co_point_a = np.where(a == 0)[0][co_point]
            co_point_b = np.where(b == 0)[0][co_point]
            a_new = np.concatenate((a[:co_point_a], np.array([0]), b[co_point_b + 1:]))
            b_new = np.concatenate((b[:co_point_b], np.array([0]), a[co_point_a + 1:]))

            all_customers = self.data['demand'].sort_values('TimeWindow').index
            for old_chromosome, chromosome in zip([a, b], [a_new, b_new]):
                custs_only = np.concatenate(split_at_delimiter(chromosome))[::2]
                missing_customers = [c for c in all_customers if c not in custs_only]
                extra_custs = [item for item, count in Counter(custs_only).items() if (count > 1 and item != 0)]

                for extra_cust in extra_custs:
                    idx = np.where(chromosome == extra_cust)[0][-1]
                    if len(missing_customers) > 0:
                        fill_cust = missing_customers.pop()
                        idx_old = np.where(old_chromosome == fill_cust)[0][-1]
                        chromosome[idx: idx + 1] = old_chromosome[idx_old: idx_old + 1]
                    else:
                        if idx >= len(chromosome) - 1:
                            print(idx, len(chromosome))
                        chromosome = np.delete(chromosome, (idx, idx+1))
            assert len(a) == ((len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1)
            assert len(b) == ((len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1)
            assert sum(a == 0) == self.data['n_vehicles'] - 1
            assert sum(b == 0) == self.data['n_vehicles'] - 1

            for veh_route in split_at_delimiter(a):
                assert (len(veh_route) % 2) == 0
            for veh_route in split_at_delimiter(b):
                assert (len(veh_route) % 2) == 0

            Y[0, k], Y[1, k] = a, b
        return Y