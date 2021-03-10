import copy
import numpy as np
import random

from pymoo.model.crossover import Crossover


class SinglePoint(Crossover):
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

            co_point = random.randint(2, len(a) - 2)
            a[co_point:], b[co_point:] = copy.deepcopy(b[co_point:]), copy.deepcopy(a[co_point:])

            all_customers = self.data['demand'].sort_values('TimeWindow').index
            for chromosome in [a, b]:
                missing_customers = [c for c in all_customers if c not in chromosome]
                extra_cust = [cust_idx for cust_idx in range(co_point, len(chromosome)) if (chromosome[cust_idx] in chromosome[:co_point] and chromosome[cust_idx] != 0)]
                extra_zeros = []
                if sum(chromosome == 0) > self.data['n_vehicles'] - 1:
                    num_extra_zeros = sum(chromosome == 0) - (self.data['n_vehicles'] - 1)
                    extra_zeros = [zero_idx for zero_idx in np.where(chromosome == 0)[0] if zero_idx >= co_point][:num_extra_zeros]
                if sum(chromosome == 0) < self.data['n_vehicles'] - 1:
                    missing_customers += [0] * (self.data['n_vehicles'] - sum(chromosome == 0) - 1)
                to_fill = np.append(extra_cust, extra_zeros)
                assert len(missing_customers) == len(to_fill)
                for entry, to_idx in zip(missing_customers, to_fill):
                    chromosome[int(to_idx)] = entry
            assert sum(a == 0) == self.data['n_vehicles'] - 1
            assert sum(b == 0) == self.data['n_vehicles'] - 1
            Y[0, k], Y[1, k] = a, b
        return Y