from pymoo.model.problem import Problem
from utils import *

class MyProblem(Problem):
    def __init__(self, data):
        self.data = data
        chromosome_len = (len(self.data['demand']) * 2) + self.data['n_vehicles'] - 1
        super().__init__(n_var=chromosome_len,  # Number of genes
                         n_obj=5,  # Number of objectives
                         type_var=np.int,
                         elementwise_evaluation=True,
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = 0  # Travel distance
        f2 = 0  # Travel time
        f3 = 0  # Time window violation
        f4 = 0  # Slack time
        f5 = 0  # Number of vehicles

        # Separate chromosome into routes by the delimiter (0)
        veh_routes = split_at_delimiter(X)
        f5 += len(veh_routes)

        # Loop over each route
        for veh_i, veh_route in enumerate(veh_routes):
            # Initial pick-up time assumes picking up first customer on time (at their time window start)
            pickup_time = self._get_time_window(customer=veh_route[0], bound='start')
            ss_node = self.data['service_stations'].iloc[veh_i].NodeID

            for i in range(0, len(veh_route), 2):
                assert len(veh_route) % 2 == 0
                customer = veh_route[i]  # Customer being serviced
                travel_time = 0  # Travel time of the one route

                customer_org = self.data['demand'].loc[customer]['OriginNodeID']
                customer_dest = self.data['demand'].loc[customer]['DestinationNodeID']

                # Add travel time and distance to pick up customer and drop them off
                travel_time += (self.data['travel_time'][ss_node][customer_org] + self.data['travel_time'][customer_org][customer_dest])
                f1 += (self.data['travel_distance'][ss_node][customer_org] + self.data['travel_distance'][customer_org][customer_dest])

                ss_node = self.data['service_stations'].loc[veh_route[i + 1]].NodeID

                # Add travel time and distance from customer destination to service station destination
                travel_time += self.data['travel_time'][customer_dest][ss_node]
                f1 += self.data['travel_distance'][customer_dest][ss_node]

                # Find time window violation
                if i != 0:
                    pickup_time = pickup_time + timedelta(minutes=travel_time)
                    if pickup_time < self._get_time_window(customer, bound='start'):
                        slack = (self._get_time_window(customer, bound='start') - pickup_time)
                        f4 += slack.seconds / 60
                        pickup_time = self._get_time_window(customer, bound='start')
                    elif pickup_time > self._get_time_window(customer, bound='end'):
                        time_violation = (pickup_time - self._get_time_window(customer, bound='end'))
                        f3 += time_violation.seconds / 60

                f2 += travel_time
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