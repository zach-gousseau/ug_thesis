import numpy as np
import random
import copy

from pymoo.model.sampling import Sampling
from utils import *


class RandomSample(Sampling):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def make_assignments(self):
        veh_assignment = {ss_id: [] for ss_id in self.data['service_stations'].index}
        bus_riders = {}
        ss_assignment = {}

        for customer in self.data['customers']:
            veh_choice = random.choice(self.data['service_stations'].index)
            veh_assignment[veh_choice].append(customer)
            if random.random() < 0.2:
                org_ss_choice = random.choice(self.data['service_stations'].index)
                dst_ss_choice = random.choice(self.data['service_stations'].index)
                veh_choice_after_bus = random.choice(self.data['service_stations'].index)
                bus_choice_on = random.choice(self.data['bus_stops'])
                bus_choice_off = random.choice(self.data['bus_stops'])
                while bus_choice_off == bus_choice_on:
                    bus_choice_off = random.choice(self.data['bus_stops'])
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
        return veh_assignment, bus_riders, ss_assignment

    def build_chromosome(self, veh_assignment, bus_riders, ss_assignment):
        chromosome = ([], [], [], [])
        for veh in veh_assignment:
            if chromosome[0]:
                chromosome = append_2d(chromosome, 0)

            customers = veh_assignment[veh]
            for customer in customers:

                if (customer in bus_riders) and not (
                        bus_riders[customer]['off_done'] and bus_riders[customer]['on_done']):

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

        return np.array(chromosome)

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), 0, dtype=np.object)
        for i in range(n_samples):

            # Choose vehicle, bus, ss for each customer
            veh_assignment, bus_riders, ss_assignment = self.make_assignments()

            # Build the chromosome
            chromosome = self.build_chromosome(veh_assignment, bus_riders, ss_assignment)

            # Checks
            for customer in self.data['customers']:
                assert len(np.where(chromosome[0] == customer)[0]) <= 2
                assert customer in chromosome[0]
            X[i, 0] = chromosome
        return X


class BetterSample(Sampling):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def cluster(self):
        veh_assignment = {ss_id: [] for ss_id in self.data['service_stations'].index}
        for customer in self.data['customers']:
            org = self.data['demand']['OriginNodeID'][customer]
            dst = self.data['demand']['DestinationNodeID'][customer]

            # Find total distances from SS -> Customer -> Destination for each SS
            ss_distances = np.array(
                [self.data['travel_distance'][org][ss] + self.data['travel_distance'][ss][dst] for ss in
                 self.data['service_stations'].NodeID]
            )

            # Sort the distances and choose a (weighted) random choice of the closest 1/3 SS's.
            sorted_ss = self.data['service_stations'].index[ss_distances.argsort()]
            sorted_ss = sorted_ss[:int(np.ceil(len(sorted_ss)/3))]
            weights = reversed(range(1, len(sorted_ss) + 1))
            weights = [weight / sum(weights) for weight in weights]
            ss_id = np.random.choice(sorted_ss, p=weights)

            # ss_id = self.data['service_stations'].index[np.argmin(
            # [self.data['travel_distance'][org][ss] + self.data['travel_distance'][ss][dst]
            #  for ss in self.data['service_stations'].NodeID])]

            veh_assignment[ss_id].append(customer)
        return veh_assignment

    def find_nearest_ss_random(self, node, proportion=0.33):
        # Find distances from Destination -> SS
        ss_distances = np.array([self.data['travel_distance'][node][ss] for ss in self.data['service_stations'].NodeID])

        # Sort the distances and choose a (weighted) random choice of the closest 1/3 SS's.
        sorted_ss = self.data['service_stations'].index[ss_distances.argsort()]
        sorted_ss = sorted_ss[:int(np.ceil(len(sorted_ss) * proportion))]
        weights = list(reversed(range(1, len(sorted_ss) + 1)))
        weights = [weight / sum(weights) for weight in weights]
        ss_choice = np.random.choice(sorted_ss, p=weights)
        return ss_choice

    def make_assignments(self):
        veh_assignment = {ss_id: [] for ss_id in self.data['service_stations'].index}
        bus_riders = {}
        ss_assignment = {}

        largest_distance = max([max([dic[node] for node in self.data['travel_distance']]) for dic in self.data['travel_distance'].values()])

        for customer in self.data['customers']:
            org = self.data['demand']['OriginNodeID'][customer]
            dst = self.data['demand']['DestinationNodeID'][customer]

            # Find total distances from SS -> Customer -> Destination for each SS
            ss_distances = np.array(
                [self.data['travel_distance'][org][ss] + self.data['travel_distance'][ss][dst] for ss in
                 self.data['service_stations'].NodeID]
            )

            # Sort the distances and choose a (weighted) random choice of the closest 1/3 SS's.
            sorted_ss = self.data['service_stations'].index[ss_distances.argsort()]
            sorted_ss = sorted_ss[:int(np.ceil(len(sorted_ss) * 0.33))]
            weights = list(reversed(range(1, len(sorted_ss) + 1)))
            weights = [weight / sum(weights) for weight in weights]
            veh_choice = np.random.choice(sorted_ss, p=weights)
            veh_assignment[veh_choice].append(customer)

            # Randomly select whether they take the bus based on the distance between origin and destination
            bus_prob = (1 - self.data['travel_distance'][org][dst] / largest_distance) * 0.5  # Weighted choice by distance, scale to some max probability (0.5)
            if random.random() < bus_prob:
                org_ss_choice = self.find_nearest_ss_random(org)  # SS after being dropped off at bus stop

                bus_choice_on = random.choice(self.data['bus_stops'])  # Choice of bus stop to board
                bus_choice_off = random.choice(self.data['bus_stops'])   # Choice of bus stop to get off
                while bus_choice_off == bus_choice_on:
                    bus_choice_off = random.choice(self.data['bus_stops'])

                veh_choice_after_bus = random.choice(self.data['service_stations'].index)  # Vehicle picking up the customer from the bus stop
                dst_ss_choice = self.find_nearest_ss_random(dst)  # SS after being picked up from the bus stop

                # Store choices
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

                # Add customer to the list of customers to be picked up by the post-bus vehicle
                veh_assignment[veh_choice_after_bus].append(customer)
            else:
                # Choose one of the nearest SSs
                ss_assignment[customer] = self.find_nearest_ss_random(dst)

        return veh_assignment, bus_riders, ss_assignment

    def make_assignments_greedy(self):
        bus_riders = {}
        ss_assignment = {}

        largest_distance = max([max([dic[node] for node in self.data['travel_distance']]) for dic in self.data['travel_distance'].values()])

        customers = self.find_greedy_path(copy.deepcopy(self.data['customers']))
        veh_routes = np.array_split(customers, len(self.data['service_stations']))
        veh_assignment = {ss_id: veh_route for ss_id, veh_route in zip(self.data['service_stations'].index, veh_routes)}


        for customer in self.data['customers']:
            org = self.data['demand']['OriginNodeID'][customer]
            dst = self.data['demand']['DestinationNodeID'][customer]

            veh_choice = None
            for veh in veh_assignment:
                if customer in veh_assignment[veh]:
                    veh_choice = veh
                    break

            assert veh_choice is not None

            # Randomly select whether they take the bus based on the distance between origin and destination
            bus_prob = (1 - self.data['travel_distance'][org][dst] / largest_distance) * 0.5  # Weighted choice by distance, scale to some max probability (0.5)
            if random.random() < bus_prob:
                org_ss_choice = self.find_nearest_ss_random(org)  # SS after being dropped off at bus stop

                bus_choice_on = random.choice(self.data['bus_stops'])  # Choice of bus stop to board
                bus_choice_off = random.choice(self.data['bus_stops'])   # Choice of bus stop to get off
                while bus_choice_off == bus_choice_on:
                    bus_choice_off = random.choice(self.data['bus_stops'])

                veh_choice_after_bus = random.choice(self.data['service_stations'].index)  # Vehicle picking up the customer from the bus stop
                dst_ss_choice = self.find_nearest_ss_random(dst)  # SS after being picked up from the bus stop

                # Store choices
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

                # Add customer to the list of customers to be picked up by the post-bus vehicle
                idx = random.randint(0, len(veh_assignment[veh_choice_after_bus]))
                veh_assignment[veh_choice_after_bus] = np.insert(veh_assignment[veh_choice_after_bus], idx, customer)
            else:
                # Choose one of the nearest SSs
                ss_assignment[customer] = self.find_nearest_ss_random(dst)

        return veh_assignment, bus_riders, ss_assignment

    def find_greedy_path(self, customers):
        if not customers:
            return customers
        greedy_path = np.full_like(customers, 0)

        cur = random.choice(customers)
        customers.remove(cur)
        greedy_path[0] = cur

        i = 0
        while customers:
            i += 1
            min_cust = None
            min_dist = np.inf
            for c in customers:
                dist = self.data['travel_distance'][self.data['demand']['OriginNodeID'][cur]][self.data['demand']['OriginNodeID'][c]]
                if dist < min_dist and c != cur:
                    min_dist = dist
                    min_cust = c
            if min_cust is None:
                assert len(customers) == 1
                min_cust = customers[0]

            greedy_path[i] = min_cust
            customers.remove(min_cust)
            cur = min_cust
        return greedy_path

    def build_chromosome(self, veh_assignment, bus_riders, ss_assignment):
        chromosome = ([], [], [], [])
        for veh in veh_assignment:
            if chromosome[0]:
                chromosome = append_2d(chromosome, 0)

            customers = veh_assignment[veh]

            # GET GREEDY PATH
            # customers = self.find_greedy_path(customers)

            for customer in customers:

                if (customer in bus_riders) and not (
                        bus_riders[customer]['off_done'] and bus_riders[customer]['on_done']):

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

        return np.array(chromosome)

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), 0, dtype=np.object)
        for i in range(n_samples):

            # Choose vehicle, bus, ss for each customer
            veh_assignment, bus_riders, ss_assignment = self.make_assignments_greedy()

            # Build the chromosome
            chromosome = self.build_chromosome(veh_assignment, bus_riders, ss_assignment)

            # Checks
            for customer in self.data['customers']:
                assert len(np.where(chromosome[0] == customer)[0]) <= 2
                assert customer in chromosome[0]
            X[i, 0] = chromosome
        return X
