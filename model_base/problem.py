import numpy as np
from pymoo.model.problem import Problem
from datetime import datetime, date, timedelta
from utils import *

class MyProblem(Problem):
    def __init__(self, travel_time, travel_distance, service_stations, demand, time_windows, n_vehicles):
        self.travel_time = travel_time
        self.travel_distance = travel_distance
        self.service_stations = service_stations
        self.demand = demand
        self.time_windows = time_windows
        self.n_vehicles = n_vehicles

        chromosome_len = len(self.demand) + self.n_vehicles - 1

        super().__init__(n_var=chromosome_len,  # Number of genes
                         n_obj=3,  # Number of objectives
                         type_var=np.int,
                         elementwise_evaluation=True,
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = 0  # Travel distance
        f2 = 0  # Travel time
        f3 = 0  # Time window violation

        # Separate chromosome into routes by the delimiter (0)
        veh_routes = split_at_delimiter(X)

        # Loop over each route
        for veh_i, veh_route in enumerate(veh_routes):
            # Initial pick-up time assumes picking up first customer on time (at their time window start)
            pickup_time = self._get_time_window(customer=veh_route[0], bound='start')
            ss_node = self.service_stations.iloc[veh_i].NodeID

            for i in range(len(veh_route)):
                customer = veh_route[i]  # Customer being serviced
                travel_time = 0  # Travel time of the one route

                customer_org = self.demand.loc[customer]['OriginNodeID']
                customer_dest = self.demand.loc[customer]['DestinationNodeID']

                # Add travel time and distance to pick up customer and drop them off
                travel_time += (self.travel_time[ss_node][customer_org] + self.travel_time[customer_org][customer_dest])
                f1 += (self.travel_distance[ss_node][customer_org] + self.travel_distance[customer_org][customer_dest])

                # Find best service station destination
                if len(veh_route) > 1 and i != len(veh_route) - 1:
                    ss_node = self.service_stations.index[np.argmin(
                        [self.travel_distance[customer_dest][ss] + self.travel_distance[ss][veh_route[i + 1]]
                         for ss in self.service_stations.index])]
                else:
                    ss_node = self.service_stations.index[np.argmin([self.travel_distance[customer_dest][ss]
                         for ss in self.service_stations.index])]

                # Add travel time and distance from customer destination to service station destination
                travel_time += self.travel_time[customer_dest][ss_node]
                f1 += self.travel_distance[customer_dest][ss_node]

                # Find time window violation
                if i != 0:
                    pickup_time = pickup_time + timedelta(minutes=travel_time)
                    if pickup_time < self._get_time_window(customer, bound='start'):
                        pickup_time = self._get_time_window(customer, bound='start')
                    elif pickup_time > self._get_time_window(customer, bound='end'):
                        time_violation = (pickup_time - self._get_time_window(customer, bound='end'))
                        f3 += time_violation.seconds / 60

                f2 += travel_time

        out["F"] = np.array([f1, f2, f3], dtype=np.float)

    def _get_time_window(self, customer=None, bound='start'):
        if bound == 'start':
            bound = 'StartTime'
        elif bound == 'end':
            bound = 'EndTime'
        else:
            raise ValueError('Bound must be either start or end.')
        if customer is not None:
            return self.time_windows.loc[self.demand.loc[customer]['TimeWindow']][bound]