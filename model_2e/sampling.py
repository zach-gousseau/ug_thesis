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
        X = np.full((n_samples, 1), 0, dtype=np.int)
        veh_assignment = {ss_id: [] for ss_id in self.data['service_stations'].index}

        for i in range(n_samples):
            for customer in self.data['demand'].index:
                ss_choice = random.choice(self.data['service_stations'].index)
                veh_assignment[ss_choice] = customer
                if random.random > 0.2:
                    bus_choice_on = random.choice(self.data['bus_stops'])
                    for bus in self.data['bus_assignment']:
                        if bus_choice_on in self.data['bus_assignment'][bus]:
                            bus_choice_off = self.data['bus_assignment'][bus]
                            while bus_choice_off == bus_choice_on:
                                bus_choice_off = self.data['bus_assignment'][bus]
            # TODO
            for veh in veh_assignment:

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



class SmartSortSample(Sampling):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def cluster(self):
        veh_assignment = {ss_id: [] for ss_id in self.data['service_stations'].index}
        for customer in self.data['demand'].index:
            org = self.data['demand'].loc[customer]['OriginNodeID']
            dst = self.data['demand'].loc[customer]['DestinationNodeID']

            # Find total distances from SS -> Customer -> Destination for each SS
            ss_distances = np.array(
                [self.data['travel_distance'][org][ss] + self.data['travel_distance'][ss][dst] for ss in self.data['service_stations'].NodeID]
            )

            # Sort the distances and choose a (weighted) random choice of the closest 2 SS.
            sorted_ss = self.data['service_stations'].index[ss_distances.argsort()]
            sorted_ss = sorted_ss[:2]
            weights = [2, 1]
            weights = [weight/sum(weights) for weight in weights]
            ss_id = np.random.choice(sorted_ss, p=weights)

            # ss_id = self.data['service_stations'].index[np.argmin(
                # [self.data['travel_distance'][org][ss] + self.data['travel_distance'][ss][dst]
                #  for ss in self.data['service_stations'].NodeID])]

            veh_assignment[ss_id].append(customer)
        return veh_assignment

    def _do(self, problem, n_samples, **kwargs):
        chromosome_len = (len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1
        X = np.full((n_samples, chromosome_len), 0, dtype=np.int)
        for i in range(n_samples):

            # Get vehicle assignment
            veh_assignment = self.cluster()
            sorted_customers = self.data['demand'].sort_values('TimeWindow').index
            j = 0
            for veh in veh_assignment:
                veh_assignment[veh] = sorted_customers[sorted_customers.isin(veh_assignment[veh])]
                for k in range(len(veh_assignment[veh])):
                    cust = veh_assignment[veh][k]
                    customer_dest = self.data['demand'].loc[cust]['DestinationNodeID']

                    X[i][j] = cust  # Set customer

                    # Get best service station
                    if k != len(veh_assignment[veh]) - 1:
                        next_cust = veh_assignment[veh][k + 1]
                        ss_id = self.data['service_stations'].index[np.argmin(
                            [self.data['travel_distance'][customer_dest][self.data['service_stations'].loc[ss].NodeID]
                             + self.data['travel_distance'][self.data['service_stations'].loc[ss].NodeID][next_cust]
                             for ss in self.data['service_stations'].index])]
                    else:
                        ss_id = self.data['service_stations'].index[np.argmin(
                            [self.data['travel_distance'][customer_dest][self.data['service_stations'].loc[ss].NodeID]
                             for ss in self.data['service_stations'].index])]

                    X[i][j + 1] = ss_id
                    j += 2
                if j != len(X[i]):
                    X[i][j] = 0
                    j += 1

            assert sum(X[i] == 0) == (self.data['n_vehicles'] - 1)
        return X
