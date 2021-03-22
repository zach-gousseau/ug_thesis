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
        X = np.full((n_samples, 1), 0, dtype=np.object)
        for i in range(n_samples):

            veh_assignment = {ss_id: [] for ss_id in self.data['service_stations'].index}
            # veh_assignment_after_bus = {}
            bus_riders = {}
            ss_assignment = {}

            for customer in self.data['demand'].index:
                veh_choice = random.choice(self.data['service_stations'].index)
                veh_assignment[veh_choice].append(customer)
                if random.random() < 0.2:
                    org_ss_choice = random.choice(self.data['service_stations'].index)
                    dst_ss_choice = random.choice(self.data['service_stations'].index)
                    veh_choice_after_bus = random.choice(self.data['service_stations'].index)
                    bus_choice = random.choice(list(self.data['bus_assignment'].keys()))
                    bus_choice_on = random.choice(self.data['bus_assignment'][bus_choice])
                    bus_choice_off = random.choice(self.data['bus_assignment'][bus_choice])
                    while bus_choice_off == bus_choice_on:
                        bus_choice_off = random.choice(self.data['bus_assignment'][bus_choice])
                    bus_riders[customer] = {
                        'org_ss': org_ss_choice,  # Origin SS
                        'org_veh': veh_choice,  # Origin vehicle which drops customer off at bus station
                        'on': bus_choice_on,  # Bus stop choice for onboarding
                        'off': bus_choice_off,  # Bus stop choice for offboarding
                        'dst_ss': dst_ss_choice,  # Destination SS
                        'dst_veh': veh_choice_after_bus,  # Destination vehicle which drops customer off at their destation
                        'off_done': False,  # Helper variable for building chromosome
                        'on_done': False,  # Helper variable for building chromosome
                    }
                    veh_assignment[veh_choice_after_bus].append(customer)
                else:
                    ss_choice = random.choice(self.data['service_stations'].index)
                    ss_assignment[customer] = ss_choice

            chromosome = ([], [], [], [])
            for veh in veh_assignment:
                if chromosome[0]:
                    chromosome = append_2d(chromosome, 0)

                customers = veh_assignment[veh]
                for customer in customers:

                    if (customer in bus_riders) and not (bus_riders[customer]['off_done'] and bus_riders[customer]['on_done']):

                        if bus_riders[customer]['org_veh'] == veh and not bus_riders[customer]['on_done']:
                            # Origin drop off at bus stop
                            chromosome[0].append(customer)  # Cust
                            chromosome[1].append(bus_riders[customer]['on'])  # Bus on
                            chromosome[2].append(-1)  # No bus off
                            chromosome[3].append(bus_riders[customer]['org_ss'])  # SS

                            bus_riders[customer]['on_done'] = True

                        elif bus_riders[customer]['dst_veh'] == veh:
                            # Destination pick up from bus stop
                            chromosome[0].append(customer)  # Cust
                            chromosome[1].append(-1)  # No bus on
                            chromosome[2].append(bus_riders[customer]['off'])  # Bus off
                            chromosome[3].append(bus_riders[customer]['dst_ss'])  # SS

                            bus_riders[customer]['off_done'] = True

                    else:
                        chromosome[0].append(customer)  # Cust
                        chromosome[1].append(-1)  # No bus
                        chromosome[2].append(-1)  # No bus
                        chromosome[3].append(ss_assignment[customer])  # SS

            chromosome = np.array(chromosome)

            # Checks
            for customer in self.data['demand'].index:
                assert len(np.where(chromosome[0] == customer)[0]) <= 2
                assert customer in chromosome[0]
            X[i, 0] = chromosome

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
                [self.data['travel_distance'][org][ss] + self.data['travel_distance'][ss][dst] for ss in
                 self.data['service_stations'].NodeID]
            )

            # Sort the distances and choose a (weighted) random choice of the closest 2 SS.
            sorted_ss = self.data['service_stations'].index[ss_distances.argsort()]
            sorted_ss = sorted_ss[:2]
            weights = [2, 1]
            weights = [weight / sum(weights) for weight in weights]
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
