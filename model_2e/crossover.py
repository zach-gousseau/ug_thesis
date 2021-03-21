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
            a, b = X[0, k, 0], X[1, k, 0]

            for c in [a, b]:
                for d in self.data['demand'].index:
                    assert d in c

            if random.random() > 0.5:
                a_new, b_new = self.cross_routes(a, b)
            else:
                a_new, b_new = self.cross_bus(a, b)

            # Fix-up

            # Fix number of vehicles
            delimiters = np.where(a_new[0] == 0)[0]
            while len(delimiters > self.data['n_vehicles'] - 1):
                a_new = np.delete(a_new, delimiters[-1], axis=1)
                delimiters = np.where(a_new[0] == 0)[0]

            delimiters = np.where(b_new[0] == 0)[0]
            while len(delimiters > self.data['n_vehicles'] - 1):
                b_new = np.delete(b_new, delimiters[-1], axis=1)
                delimiters = np.where(b_new[0] == 0)[0]

            # Fix the rest...
            all_customers = self.data['demand'].sort_values('TimeWindow').index
            chromosomes = {'a': {'old': a, 'new': a_new},
                           'b': {'old': b, 'new': b_new}}
            for chromosome_name in chromosomes:
                old_chromosome = chromosomes[chromosome_name]['old']
                chromosome = chromosomes[chromosome_name]['new']

                missing_customers = [c for c in all_customers if c not in chromosome[0]]
                extra_customers = [item for item, count in Counter(chromosome[0]).items() if (count > 1 and item != 0)]

                for extra_customer in extra_customers:
                    idx = np.where(chromosome[0] == extra_customer)[0]
                    strict = True
                    while len(idx) > 2:
                        # Try once to replace one customer that has no bus, then just remove an arbitrary one.
                        found = False
                        for i in idx:
                            if chromosome[1][i] == -1 and chromosome[2][i] == -1 or not strict:
                                if len(missing_customers) > 0:
                                    # Can be replaced by a missing customer
                                    fill_cust = missing_customers.pop()
                                    i_old = np.where(old_chromosome[0] == fill_cust)[0][-1]
                                    chromosome[0][i] = fill_cust
                                    chromosome[3][i] = old_chromosome[3][i_old]
                                    found = True
                                    break
                                else:
                                    # Needs to just be deleted.
                                    chromosome = np.delete(chromosome, i, axis=1)
                                    found = True
                                    break
                        if not found:
                            strict = False
                        else:
                            strict = True
                        idx = np.where(chromosome[0] == extra_customer)[0]
                    assert extra_customer in chromosome[0]

                    if len(idx) == 1:
                        # Looks good. This is no longer an extra customer.
                        break

                    elif len(idx) == 2:
                        # Find corresponding bus stops in case this is a bus rider
                        if chromosome[1][idx[0]] != -1 and chromosome[2][idx[0]] == -1:
                            bus_on_idx = idx[0]
                            bus_off_idx = idx[1]
                        elif chromosome[1][idx[1]] != -1 and chromosome[2][idx[1]] == -1:
                            bus_on_idx = idx[1]
                            bus_off_idx = idx[0]
                        else:
                            bus_on_idx, bus_off_idx = None, None

                        if bus_off_idx is not None and bus_on_idx is not None:
                            bus_on = chromosome[1][bus_on_idx]
                            bus_off = chromosome[2][bus_off_idx]

                        if bus_off_idx is None and bus_on_idx is None:
                            # This is an extra customer, not a bus rider.
                            chromosome[1][idx[0]] = -1
                            chromosome[2][idx[0]] = -1

                            chromosome = np.delete(chromosome, idx[1], axis=1)

                        elif bus_off == -1 and bus_on != -1:
                            # Missing bus_off. Take from old chromosome if it exists. Else remove.
                            for old_chromosome_0 in [a, b]:
                                jdx = np.where(old_chromosome_0[0] == extra_customer)[0]
                                assert len(jdx) <= 2
                                if len(jdx) == 2:
                                    for j in jdx:
                                        if old_chromosome_0[2][j] != -1:
                                            chromosome[2][bus_off_idx] = old_chromosome_0[2][j]

                            if chromosome[2][bus_off_idx] == -1:
                                # Not found therefore remove existing bus_on
                                chromosome = np.delete(chromosome, bus_on_idx, axis=1)

                        elif bus_off != -1 and bus_on == -1:
                            # Missing bus_on. Take from old chromosome if it exists. Else remove
                            for old_chromosome_0 in [a, b]:
                                jdx = np.where(old_chromosome_0[0] == extra_customer)[0]
                                assert jdx <= 2
                                if len(jdx) == 2:
                                    for j in jdx:
                                        if chromosome[1][j] != -1:
                                            chromosome[1][bus_on_idx] = chromosome[1][j]

                            if chromosome[1][bus_on_idx] == -1:
                                # Not found therefore remove existing bus_off
                                chromosome = np.delete(chromosome, bus_off_idx, axis=1)

                if missing_customers:
                    for missing_customer in missing_customers:
                        # Can be replaced by a missing customer
                        i_old = np.where(old_chromosome[0] == missing_customer)[0][-1]
                        insert_point = random.randint(0, len(chromosome[0]))  # Choose random insert point
                        chromosome = np.insert(chromosome, insert_point, (missing_customer, -1, -1, old_chromosome[3][i_old]), axis=1)

                # Checks
                for customer in self.data['demand'].index:
                    assert customer in chromosome[0]

                assert len(np.where(chromosome[0] == 0)[0]) <= self.data['n_vehicles']

                chromosomes[chromosome_name]['new'] = chromosome

            a_new, b_new = chromosomes['a']['new'], chromosomes['b']['new']
            Y[0, k, 0], Y[1, k, 0] = a_new, b_new
        return Y

    def cross_routes(self, a, b):
        a_new, b_new = [[], [], [], []], [[], [], [], []]
        routes_a = split_at_delimiter_2d(a)
        routes_b = split_at_delimiter_2d(b)

        co_point_a = random.randint(1, len(routes_a))
        co_point_b = random.randint(1, len(routes_b))

        a_new = join_with_delimiter_2d(routes_a[:co_point_a] + routes_b[co_point_b:])
        b_new = join_with_delimiter_2d(routes_b[:co_point_b] + routes_a[co_point_a:])
        return np.stack(a_new), np.stack(b_new)

    def cross_within_routes(self, a, b):
        a_new, b_new = ([], [], [], []), ([], [], [], [])
        routes_a = split_at_delimiter_2d(a)
        routes_b = split_at_delimiter_2d(b)
        for route_a, route_b in zip(routes_a, routes_b):
            if a_new[0]:
                a_new = append_2d(a_new, 0)
                b_new = append_2d(b_new, 0)

            co_point_a = random.randint(1, len(route_a[0]))
            co_point_b = random.randint(1, len(route_b[0]))

            for i in range(len(a_new)):
                a_new[i] += list(np.concatenate((route_a[i][:co_point_a], route_b[i][co_point_b:])))
                b_new[i] += list(np.concatenate((route_b[i][:co_point_b], route_a[i][co_point_a:])))
        return np.stack(a_new), np.stack(b_new)

    def cross_bus(self, a, b):
        # a_new, b_new = copy.deepcopy(a), copy.deepcopy(b)
        #
        # buses_idx_a = np.where(np.logical_and(a > 500, a < 600))
        # buses_idx_b = np.where(np.logical_and(b > 500, b < 600))
        #
        # co_point = random.randint(0, min(len(buses_idx_a), len(buses_idx_a)))
        # genes_to_swap_a = buses_idx_a[:-co_point]
        # genes_to_swap_b = buses_idx_b[:-co_point]
        #
        # for i in range(len(genes_to_swap_a)):
        #     a_new[genes_to_swap_a[i]], b_new[genes_to_swap_b[i]] = b_new[genes_to_swap_b[i]], a_new[genes_to_swap_a[i]]
        # return a_new, b_new
        return a, b

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


            Y[0, k], Y[1, k] = a, b
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