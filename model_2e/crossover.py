import copy
import numpy as np
import pandas as pd
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

            choice = random.choice([1, 2, 3, 4])
            if choice == 1:
                a_new, b_new = self.cross_routes(a, b)
            elif choice == 2:
                a_new, b_new = self.cross_within_routes(a, b)
            elif choice == 3:
                a_new, b_new = self.cross_bus(a, b)
            elif choice == 4:
                a_new, b_new = self.cross_ss(a, b)

            # Fix-up
            chromosomes = self.fix_up(a, b, a_new, b_new)

            a_new, b_new = chromosomes['a']['new'], chromosomes['b']['new']
            for a in a_new, b_new:
                bus_riders = list(set(a[0][np.logical_or(a[1] > 0, a[2] > 0)]))
                for b in bus_riders:
                    assert len(np.where(a[0] == b)[0]) == 2

            Y[0, k, 0], Y[1, k, 0] = a_new, b_new
        return Y

    def fix_up(self, a, b, a_new, b_new):
        # Fix number of vehicles
        delimiters = np.where(a_new[0] == 0)[0]
        while len(delimiters) > self.data['n_vehicles'] - 1:
            a_new = np.delete(a_new, delimiters[-1], axis=1)
            delimiters = np.where(a_new[0] == 0)[0]

        delimiters = np.where(b_new[0] == 0)[0]
        while len(delimiters) > self.data['n_vehicles'] - 1:
            b_new = np.delete(b_new, delimiters[-1], axis=1)
            delimiters = np.where(b_new[0] == 0)[0]

        # Fix the rest...
        all_customers = pd.DataFrame.from_dict(self.data['demand']).sort_values('TimeWindow').index
        chromosomes = {'a': {'old': a, 'new': a_new},
                       'b': {'old': b, 'new': b_new}}

        for chromosome_name in chromosomes:
            old_chromosome = chromosomes[chromosome_name]['old']
            chromosome = chromosomes[chromosome_name]['new']

            missing_customers = [c for c in all_customers if c not in chromosome[0]]

            bus_riders = list(set(chromosome[0][np.logical_or(chromosome[1] > 0, chromosome[2] > 0)]))
            for bus_rider in bus_riders:
                idx = np.where(chromosome[0] == bus_rider)[0]

                # If more than 2 entries, remove until only 2 are left
                while len(idx) > 2:
                    # Try once to replace one customer that has no bus, then just remove an arbitrary one.
                    found = False
                    for i in idx:
                        if chromosome[1][i] == -1 and chromosome[2][i] == -1:
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
                        chromosome = np.delete(chromosome, idx[-1], axis=1)

                    idx = np.where(chromosome[0] == bus_rider)[0]

                # If bus rider only has one entry, add another. Use the bus_off/bus_on and ss from its entry in either
                # of the old chromosomes.
                if len(idx) == 1:
                    level = 2 if chromosome[1][idx] != -1 else 1
                    existing_stop = chromosome[1][idx] if chromosome[1][idx] != -1 else chromosome[2][idx]

                    entry = []
                    for old_chromosome_0 in [a, b]:
                        idx_old = np.where(old_chromosome_0[0] == bus_rider)[0]
                        for i_old in idx_old:
                            if old_chromosome_0[level][i_old] != -1:

                                entry = [bus_rider, -1, -1, old_chromosome_0[3][i_old]]
                                entry[level] = old_chromosome_0[level][i_old]
                                break
                        if entry:
                            break

                    # If none found, use random.
                    if not entry:
                        print('OH NO 1')
                        entry = [bus_rider, -1, -1, random.choice(self.data['service_stations'].index)]
                        chosen_stop = random.choice(self.data['bus_stops'])
                        while chosen_stop == existing_stop:
                            chosen_stop = random.choice(self.data['bus_stops'])
                        entry[level] = chosen_stop

                    insert_point = random.randint(0, len(chromosome[0]))  # Choose random insert point
                    chromosome = np.insert(chromosome, insert_point, entry, axis=1)

                elif len(idx) == 2:

                    # If a bus_rider customer has both entries as bus_on or both as bus_off, remove one. The corresponding
                    # bus_on or bus_off will be added in the next steps.
                    while sum(chromosome[1][np.where(chromosome[0] == bus_rider)[0]] != -1) >= 2:
                        chromosome[1][np.where(chromosome[0] == bus_rider)[0][chromosome[1][np.where(chromosome[0] == bus_rider)[0]] != -1][-1]] = -1

                    while sum(chromosome[2][np.where(chromosome[0] == bus_rider)[0]] != -1) >= 2:
                        chromosome[2][np.where(chromosome[0] == bus_rider)[0][chromosome[2][np.where(chromosome[0] == bus_rider)[0]] != -1][-1]] = -1

                    # Find corresponding bus stops
                    if chromosome[1][idx[0]] != -1 and chromosome[2][idx[0]] == -1:
                        bus_on_idx = idx[0]
                        bus_off_idx = idx[1]
                    elif chromosome[1][idx[1]] != -1 and chromosome[2][idx[1]] == -1:
                        bus_on_idx = idx[1]
                        bus_off_idx = idx[0]
                    elif chromosome[1][idx[0]] == -1 and chromosome[2][idx[0]] != -1:
                        bus_on_idx = idx[1]
                        bus_off_idx = idx[0]
                    elif chromosome[1][idx[1]] == -1 and chromosome[2][idx[1]] != -1:
                        bus_on_idx = idx[0]
                        bus_off_idx = idx[1]

                    # Get bus_on and bus_off
                    assert bus_on_idx != bus_off_idx
                    bus_on = chromosome[1][bus_on_idx]
                    bus_off = chromosome[2][bus_off_idx]
                    assert bus_on != -1 or bus_off != -1

                    # Perform fixes
                    if bus_off == -1 and bus_on != -1:
                        # Missing bus_off. Take from old chromosome if it exists. Else remove.
                        ok = False
                        for old_chromosome_0 in [a, b]:
                            jdx = np.where(old_chromosome_0[0] == bus_rider)[0]
                            assert len(jdx) <= 2
                            if len(jdx) == 2:
                                for j in jdx:
                                    if old_chromosome_0[2][j] != -1:
                                        chromosome[2][bus_off_idx] = old_chromosome_0[2][j]
                                        ok = True
                        if not ok:
                            chromosome = np.delete(chromosome, bus_on_idx, axis=1)

                    elif bus_off != -1 and bus_on == -1:
                        # Missing bus_on. Take from old chromosome if it exists. Else remove
                        ok = False
                        for old_chromosome_0 in [a, b]:
                            jdx = np.where(old_chromosome_0[0] == bus_rider)[0]
                            assert len(jdx) <= 2
                            if len(jdx) == 2:
                                for j in jdx:
                                    if old_chromosome_0[1][j] != -1:
                                        chromosome[1][bus_on_idx] = old_chromosome_0[1][j]
                                        ok = True
                        if not ok:
                            chromosome = np.delete(chromosome, bus_on_idx, axis=1)

                        assert not any(np.logical_and(chromosome[1] > 0, chromosome[2] > 0))

                elif len(idx) > 2:
                    raise ValueError('aw man')

            assert sum(chromosome[1] > 0) == sum(chromosome[2] > 0)

            extra_customers = [item for item, count in Counter(chromosome[0]).items() if (count > 1 and item != 0 and item not in bus_riders)]

            for extra_customer in extra_customers:
                idx = np.where(chromosome[0] == extra_customer)[0]
                while len(idx) >= 2:
                    chromosome = np.delete(chromosome, idx[-1], axis=1)
                    idx = np.where(chromosome[0] == extra_customer)[0]

            if missing_customers:
                for missing_customer in missing_customers:
                    # Can be replaced by a missing customer
                    i_old = np.where(old_chromosome[0] == missing_customer)[0][-1]
                    insert_point = random.randint(0, len(chromosome[0]))  # Choose random insert point
                    chromosome = np.insert(chromosome, insert_point, (missing_customer, -1, -1, old_chromosome[3][i_old]), axis=1)

            # Checks
            assert sum(chromosome[1] > 0) == sum(chromosome[2] > 0)
            assert len([item for item, count in Counter(chromosome[0]).items() if (count > 1 and item != 0 and item not in bus_riders)]) == 0
            assert not any(np.logical_and(chromosome[1] > 0, chromosome[2] > 0))
            for customer in self.data['customers']:
                assert customer in chromosome[0]

            assert len(np.where(chromosome[0] == 0)[0]) <= self.data['n_vehicles']
            assert len([item for item, count in Counter(chromosome[0]).items() if (count > 2 and item != 0)]) == 0

            # Check that no bus_rider has both bus_on or both bus_off
            for bus_rider in bus_riders:
                assert not all(chromosome[1][np.where(chromosome[0] == bus_rider)] != -1)
                assert not all(chromosome[2][np.where(chromosome[0] == bus_rider)] != -1)

            chromosomes[chromosome_name]['new'] = chromosome
        return chromosomes

    def cross_routes(self, a, b):
        # a_new, b_new = [[], [], [], []], [[], [], [], []]
        routes_a = split_at_delimiter_2d(a)
        routes_b = split_at_delimiter_2d(b)

        co_point_a = random.randint(1, len(routes_a))
        co_point_b = random.randint(1, len(routes_b))

        a_new = join_with_delimiter_2d(routes_a[:co_point_a] + routes_b[co_point_b:])
        b_new = join_with_delimiter_2d(routes_b[:co_point_b] + routes_a[co_point_a:])
        return np.stack(a_new), np.stack(b_new)

    def cross_within_routes(self, a, b):
        a_new, b_new = [[], [], [], []], [[], [], [], []]
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

    def cross_bus(self, a0, b0):
        # Switch bus riders (including location of bus stops) between the two chromosome
        a_map, b_map = {}, {}
        a, b = copy.deepcopy(a0), copy.deepcopy(b0)

        bus_riders_old = list(set(a[0][np.logical_or(a[1] > 0, a[2] > 0)]))
        for bus_rider in bus_riders_old:
            a_map[bus_rider] = {}
            a_map[bus_rider]['on'] = a[1][a[0] == bus_rider][a[1][a[0] == bus_rider] != -1][0]
            a_map[bus_rider]['off'] = a[2][a[0] == bus_rider][a[2][a[0] == bus_rider] != -1][0]

        bus_riders_old = list(set(b[0][np.logical_or(b[1] > 0, b[2] > 0)]))
        for bus_rider in bus_riders_old:
            b_map[bus_rider] = {}
            b_map[bus_rider]['on'] = b[1][b[0] == bus_rider][b[1][b[0] == bus_rider] != -1][0]
            b_map[bus_rider]['off'] = b[2][b[0] == bus_rider][b[2][b[0] == bus_rider] != -1][0]

        for customer in b[0]:
            if customer in a_map:
                idx = np.argwhere(b[0] == customer)
                if len(idx) == 2:
                    # Customer is already a bus rider; simply switch the bus stops
                    b[1][idx[0]] = a_map[customer]['on']
                    b[2][idx[0]] = -1
                    b[1][idx[1]] = -1
                    b[2][idx[1]] = a_map[customer]['off']
                else:
                    # Customer is not already a bus rider; insert at random location
                    if random.random() > 0.5:
                        b[1][idx[0]] = a_map[customer]['on']
                        ss = a[3][a[0] == customer][a[2][a[0] == customer] != -1][0]  # get service station from the old entry itself
                        entry = [customer, -1, a_map[customer]['off'], ss]
                        insert_point = random.randint(0, len(b[0]))  # Choose random insert point
                        b = np.insert(b, insert_point, entry, axis=1)
                    else:
                        b[2][idx[0]] = a_map[customer]['off']
                        ss = a[3][a[0] == customer][a[1][a[0] == customer] != -1][0]  # get service station from the old entry itself
                        entry = [customer, a_map[customer]['on'], -1, ss]
                        insert_point = random.randint(0, len(b[0]))  # Choose random insert point
                        b = np.insert(b, insert_point, entry, axis=1)

        for customer in a[0]:
            if customer in b_map:
                idx = np.argwhere(a[0] == customer)
                if len(idx) == 2:
                    # Customer is already a bus rider; simply switch the bus stops
                    a[1][idx[0]] = b_map[customer]['on']
                    a[2][idx[0]] = -1
                    a[1][idx[1]] = -1
                    a[2][idx[1]] = b_map[customer]['off']
                else:
                    # Customer is not already a bus rider; insert at random location
                    if random.random() > 0.5:
                        a[1][idx[0]] = b_map[customer]['on']
                        ss = b[3][b[0] == customer][b[2][b[0] == customer] != -1][0]  # get service station from the old entry itself
                        entry = [customer, -1, b_map[customer]['off'], ss]
                        insert_point = random.randint(0, len(a[0]))  # Choose random insert point
                        a = np.insert(a, insert_point, entry, axis=1)
                    else:
                        a[2][idx[0]] = b_map[customer]['off']
                        ss = b[3][b[0] == customer][b[1][b[0] == customer] != -1][0]  # get service station from the old entry itself
                        entry = [customer, b_map[customer]['on'], -1, ss]
                        insert_point = random.randint(0, len(a[0]))  # Choose random insert point
                        a = np.insert(a, insert_point, entry, axis=1)
        assert not any(np.logical_and(a[1] > 0, a[2] > 0))
        assert not any(np.logical_and(b[1] > 0, b[2] > 0))
        assert sum(a[1] > 0) == sum(a[2] > 0)
        assert sum(b[1] > 0) == sum(b[2] > 0)
        return a, b

    def cross_ss(self, a, b):
        a_ss = {a[0][i]: a[3][i] for i in range(len(a[0]))}
        b_ss = {b[0][i]: b[3][i] for i in range(len(b[0]))}
        a[3] = np.array([b_ss[customer] for customer in a[0]])
        b[3] = np.array([a_ss[customer] for customer in b[0]])
        return a, b

# class CrossRoutes(Crossover):
#     def __init__(self, data):
#         # define the crossover: number of parents and number of offsprings
#         super().__init__(2, 2)
#         self.data = data
#
#     def _do(self, problem, X, **kwargs):
#         # The input of has the following shape (n_parents, n_matings, n_var)
#         _, n_matings, n_var = X.shape
#
#         # The output with the shape (n_offsprings, n_matings, n_var)
#         # Because there the number of parents and offsprings are equal it keeps the shape of X
#         Y = np.full_like(X, None, dtype=np.object)
#
#         # for each mating provided
#         for k in range(n_matings):
#             # get the first and the second parent
#             a, b = X[0, k], X[1, k]
#             a_new, b_new = [], []
#
#             # Perform cross-over
#             routes_a = split_at_delimiter(a)
#             routes_b = split_at_delimiter(b)
#             for route_a, route_b in zip(routes_a, routes_b):
#                 if a_new:
#                     a_new.append(0)
#                     b_new.append(0)
#                 co_point_a = random.randint(0, len(route_a) / 2 - 1) * 2
#                 co_point_b = random.randint(0, len(route_b) / 2 - 1) * 2
#                 route_a_new = np.concatenate((route_a[:co_point_a], route_b[co_point_b:]))
#                 route_b_new = np.concatenate((route_b[:co_point_b], route_a[co_point_a:]))
#                 a_new += list(route_a_new)
#                 b_new += list(route_b_new)
#
#             # Fix-up
#             all_customers = self.data['demand'].sort_values('TimeWindow').index
#             for old_chromosome, chromosome in zip([a, b], [a_new, b_new]):
#                 custs_only = np.concatenate(split_at_delimiter(chromosome))[::2]
#                 missing_customers = [c for c in all_customers if c not in custs_only]
#                 extra_custs = [item for item, count in Counter(custs_only).items() if (count > 1 and item != 0)]
#
#                 for extra_cust in extra_custs:
#                     idx = np.where(chromosome == extra_cust)[0][-1]
#                     if len(missing_customers) > 0:
#                         fill_cust = missing_customers.pop()
#                         idx_old = np.where(old_chromosome == fill_cust)[0][-1]
#                         chromosome[idx: idx + 1] = old_chromosome[idx_old: idx_old + 1]
#                     else:
#                         chromosome = np.delete(chromosome, (idx, idx+1))
#
#             # Perform checks
#             for chromosome in [a, b]:
#                 assert len(chromosome) == ((len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1)
#                 assert sum(chromosome == 0) == self.data['n_vehicles'] - 1
#                 for veh_route in split_at_delimiter(chromosome):
#                     assert (len(veh_route) % 2) == 0
#
#
#             Y[0, k], Y[1, k] = a, b
#         return Y
#
#
# class CrossAssignment(Crossover):
#     def __init__(self, data):
#         # define the crossover: number of parents and number of offsprings
#         super().__init__(2, 2)
#         self.data = data
#
#     def _do(self, problem, X, **kwargs):
#         """
#         Cuts at a route delimiter.
#         eg.
#         P1 = [A, A, A, A, 0, (B, B, B, B, 0, C, C, C, C)]
#         P2 = [D, D, D, D, 0, (E, E, E, E, 0, F, F, F, F)]
#         -->
#         C1 = [A, A, A, A, 0, (E, E, E, E, 0, F, F, F, F)]
#         C2 = [D, D, D, D, 0, (B, B, B, B, 0, C, C, C, C)]
#
#         """
#         print('doing crossover')
#         # The input of has the following shape (n_parents, n_matings, n_var)
#         _, n_matings, n_var = X.shape
#
#         # The output with the shape (n_offsprings, n_matings, n_var)
#         # Because there the number of parents and offsprings are equal it keeps the shape of X
#         Y = np.full_like(X, None, dtype=np.object)
#
#         # for each mating provided
#         for k in range(n_matings):
#
#             # get the first and the second parent
#             a, b = X[0, k], X[1, k]
#
#             co_point = random.randint(0, self.data['n_vehicles'] - 2)
#             co_point_a = np.where(a == 0)[0][co_point]
#             co_point_b = np.where(b == 0)[0][co_point]
#             a_new = np.concatenate((a[:co_point_a], np.array([0]), b[co_point_b + 1:]))
#             b_new = np.concatenate((b[:co_point_b], np.array([0]), a[co_point_a + 1:]))
#
#             all_customers = self.data['demand'].sort_values('TimeWindow').index
#             for old_chromosome, chromosome in zip([a, b], [a_new, b_new]):
#                 custs_only = np.concatenate(split_at_delimiter(chromosome))[::2]
#                 missing_customers = [c for c in all_customers if c not in custs_only]
#                 extra_custs = [item for item, count in Counter(custs_only).items() if (count > 1 and item != 0)]
#
#                 for extra_cust in extra_custs:
#                     idx = np.where(chromosome == extra_cust)[0][-1]
#                     if len(missing_customers) > 0:
#                         fill_cust = missing_customers.pop()
#                         idx_old = np.where(old_chromosome == fill_cust)[0][-1]
#                         chromosome[idx: idx + 1] = old_chromosome[idx_old: idx_old + 1]
#                     else:
#                         chromosome = np.delete(chromosome, (idx, idx+1))
#
#             # Perform checks
#             for chromosome in [a, b]:
#                 assert len(chromosome) == ((len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1)
#                 assert sum(chromosome == 0) == self.data['n_vehicles'] - 1
#                 for veh_route in split_at_delimiter(chromosome):
#                     assert (len(veh_route) % 2) == 0
#
#
#             Y[0, k], Y[1, k] = a, b
#         return Y