from pymoo.model.problem import Problem
from utils import *

class MyProblem(Problem):
    def __init__(self, data):
        self.data = data
        chromosome_len = 1
        super().__init__(n_var=chromosome_len,  # Number of genes
                         n_obj=6,  # Number of objectives
                         type_var=np.int,
                         elementwise_evaluation=True,
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = 0  # Travel distance
        f2 = 0  # Travel time
        f3 = 0  # Time window violation
        f4 = 0  # Slack time
        f5 = 0  # Number of vehicles
        f6 = 0  # Number of people on buses

        # Separate chromosome into routes by the delimiter (0)
        X = X[0]
        veh_routes = split_at_delimiter(X)
        f5 += len(veh_routes)

        # Loop over each route
        for veh_i, veh_route in enumerate(veh_routes):
            # Initial pick-up time assumes picking up first customer on time (at their time window start)
            start_time = self._get_time_window(customer=veh_route[0], bound='start')
            ss_node = self.data['service_stations'].iloc[veh_i].NodeID

            i = 0
            while i < len(veh_route):

                customer = veh_route[i]  # Customer being serviced
                travel_time = 0  # Travel time of the one route

                if veh_route[i + 2] in self.data['bus_stops']:
                    # Customer is dropped off at a bus stop
                    customer_at_destination = False
                    next_ss_node = self.data['service_stations'].loc[veh_route[i + 1]].NodeID
                    customer_org = self.data['demand'].loc[customer]['OriginNodeID']
                    customer_dest = veh_route[i + 2]
                    f6 += 1
                elif veh_route[i + 1] in self.data['bus_stops']:
                    # Customer is picked up from a bus stop
                    customer_at_destination = True
                    next_ss_node = self.data['service_stations'].loc[veh_route[i + 2]].NodeID
                    customer_org = veh_route[i + 1]
                    customer_dest = self.data['demand'].loc[customer]['DestinationNodeID']
                else:
                    # Customer is picked up and dropped off at their destination
                    customer_at_destination = True
                    next_ss_node = self.data['service_stations'].loc[veh_route[i + 1]].NodeID
                    customer_org = self.data['demand'].loc[customer]['OriginNodeID']
                    customer_dest = self.data['demand'].loc[customer]['DestinationNodeID']

                # Add travel time and distance to pick up customer and drop them off
                travel_time += (self.data['travel_time'][ss_node][customer_org] + self.data['travel_time'][customer_org][customer_dest])
                f1 += (self.data['travel_distance'][ss_node][customer_org] + self.data['travel_distance'][customer_org][customer_dest])

                # Find time window violation
                if i != 0:
                    dropoff_time = start_time + timedelta(minutes=travel_time)
                    if dropoff_time < self._get_time_window(customer, bound='start'):
                        slack = (self._get_time_window(customer, bound='start') - dropoff_time)
                        f4 += slack.seconds / 60
                        dropoff_time = self._get_time_window(customer, bound='start')
                    elif dropoff_time > self._get_time_window(customer, bound='end') and customer_at_destination:
                        time_violation = (dropoff_time - self._get_time_window(customer, bound='end'))
                        f3 += time_violation.seconds / 60
                else:
                    dropoff_time = start_time

                # Add travel time and distance from customer destination to service station destination
                travel_time += self.data['travel_time'][customer_dest][next_ss_node]
                f1 += self.data['travel_distance'][customer_dest][next_ss_node]
                f2 += travel_time

                # New start time for the next customer (drop-off time + time to get to the SS)
                start_time = dropoff_time + self.data['travel_time'][customer_dest][next_ss_node]
                ss_node = next_ss_node  # New start SS node
        out["F"] = np.array([f1, f2, f3, f4, f5], dtype=np.float)

    def _get_time_window(self, customer=None, bound='start'):
        if bound == 'start':
            bound = 'StartTime'
        elif bound == 'end':
            bound = 'EndTime'
        else:
            raise ValueError('Bound must be either start or end.')
        if customer is not None:
            return self.data['time_windows'].loc[self.data['demand'].loc[customer]['TimeWindow']][bound]
