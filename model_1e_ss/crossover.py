import copy
import numpy as np
import random
from utils import *
from collections import Counter

from pymoo.model.crossover import Crossover

class HybridCross(Crossover):
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
            if random.random() > 0.5:
                a_new, b_new = self.cross_routes(a, b)
            else:
                a_new, b_new = self.cross_assignment(a, b)

            # Fix-up
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

            # Perform checks
            for chromosome in [a, b]:
                assert len(chromosome) == ((len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1)
                assert sum(chromosome == 0) == self.data['n_vehicles'] - 1
                for veh_route in split_at_delimiter(chromosome):
                    assert (len(veh_route) % 2) == 0


            Y[0, k], Y[1, k] = a_new, b_new
        return Y

    def cross_routes(self, a, b):
        a_new, b_new = [], []
        routes_a = split_at_delimiter(a)
        routes_b = split_at_delimiter(b)
        for route_a, route_b in zip(routes_a, routes_b):
            if a_new:
                a_new.append(0)
                b_new.append(0)
            co_point_a = random.randint(0, len(route_a) / 2 - 1) * 2
            co_point_b = random.randint(0, len(route_b) / 2 - 1) * 2
            route_a_new = np.concatenate((route_a[:co_point_a], route_b[co_point_b:]))
            route_b_new = np.concatenate((route_b[:co_point_b], route_a[co_point_a:]))
            a_new += list(route_a_new)
            b_new += list(route_b_new)
        return a_new, b_new

    def cross_assignment(self, a, b):
        co_point = random.randint(0, self.data['n_vehicles'] - 2)
        co_point_a = np.where(a == 0)[0][co_point]
        co_point_b = np.where(b == 0)[0][co_point]
        a_new = np.concatenate((a[:co_point_a], np.array([0]), b[co_point_b + 1:]))
        b_new = np.concatenate((b[:co_point_b], np.array([0]), a[co_point_a + 1:]))
        return a_new, b_new

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
            a_new, b_new = [], []

            # Perform cross-over
            routes_a = split_at_delimiter(a)
            routes_b = split_at_delimiter(b)
            for route_a, route_b in zip(routes_a, routes_b):
                if a_new:
                    a_new.append(0)
                    b_new.append(0)
                co_point_a = random.randint(0, len(route_a) / 2 - 1) * 2
                co_point_b = random.randint(0, len(route_b) / 2 - 1) * 2
                route_a_new = np.concatenate((route_a[:co_point_a], route_b[co_point_b:]))
                route_b_new = np.concatenate((route_b[:co_point_b], route_a[co_point_a:]))
                a_new += list(route_a_new)
                b_new += list(route_b_new)

            # Fix-up
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

            # Perform checks
            for chromosome in [a, b]:
                assert len(chromosome) == ((len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1)
                assert sum(chromosome == 0) == self.data['n_vehicles'] - 1
                for veh_route in split_at_delimiter(chromosome):
                    assert (len(veh_route) % 2) == 0


            Y[0, k], Y[1, k] = a_new, b_new
        return Y


class CrossAssignment(Crossover):
    def __init__(self, data):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.data = data

    def _do(self, problem, X, **kwargs):
        """
        Cuts at a route delimiter.
        eg.
        P1 = [A, A, A, A, 0, (B, B, B, B, 0, C, C, C, C)]
        P2 = [D, D, D, D, 0, (E, E, E, E, 0, F, F, F, F)]
        -->
        C1 = [A, A, A, A, 0, (E, E, E, E, 0, F, F, F, F)]
        C2 = [D, D, D, D, 0, (B, B, B, B, 0, C, C, C, C)]

        """

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

            # Perform checks
            for chromosome in [a, b]:
                assert len(chromosome) == ((len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1)
                assert sum(chromosome == 0) == self.data['n_vehicles'] - 1
                for veh_route in split_at_delimiter(chromosome):
                    assert (len(veh_route) % 2) == 0


            Y[0, k], Y[1, k] = a, b
        return Y