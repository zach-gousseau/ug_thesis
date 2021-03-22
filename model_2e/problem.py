from pymoo.model.problem import Problem
from utils import *

class MyProblem(Problem):
    def __init__(self, data):
        self.data = data
        chromosome_len = 1
        super().__init__(n_var=chromosome_len,  # Number of genes
                         n_obj=6,  # Number of objectives
                         # type_var=np.object,
                         elementwise_evaluation=True,
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        X = X[0]
        f1 = 0  # Travel distance
        f2 = 0  # Travel time
        f3 = 0  # Time window violation
        # f4 = 0  # Slack time
        f5 = 0  # Number of vehicles
        f6 = 0  # Number of people on buses
        f7 = 0  # Bus timing violation

        bus_riders = {}

        # Separate chromosome into routes by the delimiter (0)
        veh_routes = split_at_delimiter_2d(X)
        f5 += len(veh_routes)

        # Loop over each route
        for veh_i, veh_route in enumerate(veh_routes):
            # Initial pick-up time assumes picking up first customer on time (at their time window start)
            start_time = self._get_time_window(customer=veh_route[0][0], bound='start')
            ss_node = self.data['service_stations'].iloc[veh_i].NodeID

            i = 0
            first_cust = True
            while i < len(veh_route[0]):

                customer = veh_route[0][i]  # Customer being serviced
                bus_on = veh_route[1][i]  # Bus on
                bus_off = veh_route[2][i]  # Bus off
                next_ss_node = self.data['service_stations'].loc[veh_route[3][i]].NodeID  # SS

                travel_time = 0  # Travel time of the one route
                if bus_on != -1 or bus_off != -1:
                    assert not (bus_on == -1 and bus_off == -1)  # Should not have both on and off.

                    if bus_on != -1:
                        # Customer is picked up from their origin and dropped off at a bus stop
                        if customer not in bus_riders:
                            bus_riders[customer] = {}
                        bus_riders[customer]['on'] = bus_on
                        customer_org = self.data['demand'].loc[customer]['OriginNodeID']
                        customer_dest = bus_on
                        f6 += 1

                    elif bus_off != -1:
                        # Customer is picked up from a bus stop and dropped off at their destination
                        if customer not in bus_riders:
                            bus_riders[customer] = {}
                        bus_riders[customer]['off'] = bus_off
                        customer_org = bus_off
                        customer_dest = self.data['demand'].loc[customer]['DestinationNodeID']

                else:
                    # Customer is picked up from their origin and dropped off at their destination
                    customer_org = self.data['demand'].loc[customer]['OriginNodeID']
                    customer_dest = self.data['demand'].loc[customer]['DestinationNodeID']

                # Add travel time and distance to pick up customer and drop them off
                travel_time += (self.data['travel_time'][ss_node][customer_org] + self.data['travel_time'][customer_org][customer_dest])
                f1 += (self.data['travel_distance'][ss_node][customer_org] + self.data['travel_distance'][customer_org][customer_dest])

                # Find time window violation
                dropoff_time = start_time + timedelta(minutes=travel_time)
                if not bus_on != -1:
                    if dropoff_time <= self._get_time_window(customer, bound='start'):
                        # slack = (self._get_time_window(customer, bound='start') - dropoff_time)
                        # f4 += slack.seconds / 60
                        dropoff_time = self._get_time_window(customer, bound='start')
                    elif dropoff_time > self._get_time_window(customer, bound='end'):
                        time_violation = (dropoff_time - self._get_time_window(customer, bound='end'))
                        f3 += time_violation.seconds / 60

                if first_cust:
                    dropoff_time = start_time
                    first_cust = False

                if bus_on != -1:
                    bus_riders[customer]['on_time'] = dropoff_time
                elif bus_off != -1:
                    bus_riders[customer]['off_time'] = dropoff_time
                    # bus_riders[customer]['off_travel_time_to_destination'] = travel_time

                # Add travel time and distance from customer destination to service station destination
                travel_time += self.data['travel_time'][customer_dest][next_ss_node]
                f1 += self.data['travel_distance'][customer_dest][next_ss_node]
                f2 += travel_time

                # New start time for the next customer (drop-off time + time to get to the SS)
                start_time = dropoff_time + timedelta(minutes=self.data['travel_time'][customer_dest][next_ss_node])
                ss_node = next_ss_node  # New start SS node
                i += 1

        # Check time window violation for bus riders
        for bus_rider in bus_riders:
            for bus in self.data['bus_assignment']:
                if bus_riders[bus_rider]['on'] in self.data['bus_assignment'][bus]:
                    break

            assert bus_riders[bus_rider]['on'] in self.data['bus_assignment'][bus]
            assert bus_riders[bus_rider]['off'] in self.data['bus_assignment'][bus]

            next_route_mask = bus_riders[bus_rider]['on_time'] < self.data['bus_schedule'][bus]['Time']

            next_day = False
            try:
                # Try finding next time bus stops at destination
                off_idx = np.where(self.data['bus_schedule'][bus]['Nodes'][next_route_mask] == bus_riders[bus_rider]['off'])[0][0]
            except IndexError:
                # If none, customer is dropped off next day
                next_route_mask = ~next_route_mask
                off_idx = np.where(self.data['bus_schedule'][bus]['Nodes'][next_route_mask] == bus_riders[bus_rider]['off'])[0][0]
                next_day = True

            off_time = self.data['bus_schedule'][bus]['Time'][next_route_mask][off_idx] + timedelta(days=next_day)

            if off_time > bus_riders[bus_rider]['off_time']:
                bus_schedule_violation = off_time - bus_riders[bus_rider]['off_time']
                f7 += bus_schedule_violation.seconds / 60

        out["F"] = np.array([f1, f2, f3, f5, f6, f7], dtype=np.float)

    def _get_time_window(self, customer=None, bound='start'):
        if bound == 'start':
            bound = 'StartTime'
        elif bound == 'end':
            bound = 'EndTime'
        else:
            raise ValueError('Bound must be either start or end.')
        if customer is not None:
            return self.data['time_windows'].loc[self.data['demand'].loc[customer]['TimeWindow']][bound]
