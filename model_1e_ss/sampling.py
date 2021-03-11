import numpy as np
import random
import copy

from pymoo.model.sampling import Sampling
from utils import *

class RandomSample(Sampling):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _do(self, problem, n_samples, **kwargs):
        chromosome_len = (len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1
        X = np.full((n_samples, chromosome_len), 0, dtype=np.int)
        for i in range(n_samples):

            # Get vehicle delimiters
            veh_delimiter = random.randint(0, len(self.data['demand']))
            veh_delimiters = []
            for _ in range(self.data['n_vehicles'] - 1):
                while veh_delimiter in veh_delimiters:
                    veh_delimiter = random.randint(0, len(self.data['demand']))
                veh_delimiters.append(veh_delimiter)

            customers = copy.deepcopy(list(self.data['demand'].index))
            # service_stations = copy.deepcopy(list(self.data['service_stations'].index))
            random.shuffle(customers)

            count, offset, j = 0, 0, 0
            while j < chromosome_len:
                if count in veh_delimiters:
                    j += 1
                    count += 1
                    continue
                X[i][j + offset] = customers.pop()
                X[i][j + offset + 1] = self.data['service_stations'].index[random.randint(0, len(self.data['service_stations'])) - 1]
                count += 1
                j += 2
            assert sum(X[i] == 0) == (self.data['n_vehicles'] - 1)
        return X

class SortSmartSample(Sampling):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _do(self, problem, n_samples, **kwargs):
        chromosome_len = (len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1
        X = np.full((n_samples, chromosome_len), 0, dtype=np.int)
        for i in range(n_samples):

            # Get vehicle delimiters
            veh_delimiter = random.randint(0, len(self.data['demand']))
            veh_delimiters = []
            for _ in range(self.data['n_vehicles'] - 1):
                while veh_delimiter in veh_delimiters:
                    veh_delimiter = random.randint(0, len(self.data['demand']))
                veh_delimiters.append(veh_delimiter)

            customers = copy.deepcopy(list(self.data['demand'].index))
            # service_stations = copy.deepcopy(list(self.data['service_stations'].index))
            random.shuffle(customers)

            count, offset, j = 0, 0, 0
            while j < chromosome_len:
                if count in veh_delimiters:
                    j += 1
                    count += 1
                    continue
                X[i][j + offset] = customers.pop()
                X[i][j + offset + 1] = -99
                j += 2
                count += 1

            ss_idx = np.where(X[i] == -99)[0]
            for j in ss_idx:

                if j != len(X[i]) - 1:
                    next_cust = X[i][j + 1]  # Next customer
                else:
                    next_cust = 0  # If at the end of the chromosome, no other customer to pick up

                customer_dest = self.data['demand'].loc[X[i][j - 1]]['DestinationNodeID']

                # Find best service station destination
                if next_cust != 0:
                    ss_node = self.data['service_stations'].index[np.argmin(
                        [self.data['travel_distance'][customer_dest][self.data['service_stations'].loc[ss].NodeID]
                         + self.data['travel_distance'][self.data['service_stations'].loc[ss].NodeID][next_cust]
                         for ss in self.data['service_stations'].index])]
                else:
                    ss_node = self.data['service_stations'].index[np.argmin([self.data['travel_distance'][customer_dest][self.data['service_stations'].loc[ss].NodeID]
                         for ss in self.data['service_stations'].index])]
                X[i][j] = ss_node

            assert sum(X[i] == 0) == (self.data['n_vehicles'] - 1)
        return X

# class OrderByTimeSample(Sampling):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data
#
#     def _do(self, problem, n_samples, **kwargs):
#         chromosome_len = len(self.data['demand']) + self.data['n_vehicles'] - 1
#         X = np.full((n_samples, chromosome_len), 0, dtype=np.int)
#         for i in range(n_samples):
#             chromosome = list(self.data['demand'].index) + [0] * (self.data['n_vehicles'] - 1)
#             random.shuffle(chromosome)
#             route_indices = split_at_delimiter(chromosome, return_idx=True)
#             for route_index in route_indices:
#                 if route_index[0] == route_index[1]:
#                     continue
#                 veh_route = chromosome[route_index[0]: route_index[1]]
#                 customers = self.data['demand'].sort_values('TimeWindow').index
#                 chromosome[route_index[0]: route_index[1]] = customers[customers.isin(veh_route)]
#             X[i] = np.array(chromosome)
#             assert sum(X[i] == 0) == (self.data['n_vehicles'] - 1)
#         return X