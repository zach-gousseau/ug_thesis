import copy
import os
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from pymoo.algorithms.nsga2 import NSGA2

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination


import pandas as pd
import random
import seaborn as sns

from itertools import groupby

from utils import *
from crossover import *
from mutation import *
from sampling import *
from problem import *

class Algorithm:
    def __init__(self):
        xls = pd.ExcelFile(r'data/FictitiousData.xlsx')
        self.demand = pd.read_excel(xls, 'CustomerDemand', header=0, index_col='CustomerID')
        self.service_stations = pd.read_excel(xls, 'ServiceStations', header=0, index_col='ID')
        self.bus_schedule = pd.read_excel(xls, 'PublicTransit', header=0)  # Bus schedule
        self.bus_assignment = self._get_bus_assignments(self.bus_schedule)  # Nodes visited by each bus

        time_windows = pd.read_excel(xls, 'TimeWindows', header=0, index_col='Code')
        time_windows['StartTime'] = [datetime.combine(date.today(), t) for t in time_windows['StartTime']]
        time_windows['EndTime'] = [datetime.combine(date.today(), t) for t in time_windows['EndTime']]
        self.time_windows = time_windows

        travel_distance = pd.read_excel(xls, 'OD-TravelDist', index_col=0, header=0)
        travel_time = pd.read_excel(xls, 'OD-TravelTime', index_col=0, header=0)
        self.travel_distance = self._od_to_dict(travel_distance)
        self.travel_time = self._od_to_dict(travel_time)

        self.n_vehicles = len(self.service_stations)

        self.result = None
        self.problem = None

    def go(self, ngen, pop_size, n_offspring):
        self.problem = MyProblem(self.travel_time, self.travel_distance, self.service_stations, self.demand, self.time_windows, self.n_vehicles)
        termination = get_termination("n_gen", ngen)
        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offspring,
            sampling=MySampling(self.demand, self.n_vehicles),
            crossover=MyCrossover(self.demand),
            mutation=MyMutation(),
        )
        # Optimize a copy of the algorithm to ensure reproducibility
        obj = copy.deepcopy(algorithm)

        obj.setup(self.problem, termination=termination, seed=1)

        # until the termination criterion has not been met
        while obj.has_next():
            obj.next()  # Performs an iteration of the algorithm

            # Print current iteration results
            print(f"gen: {obj.n_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV').min()} ideal: {obj.opt.get('F').min(axis=0)}")

        result = obj.result()  # Final results
        self.result = result

    def save_pareto(self):
        # Get the pareto-set and pareto-front for plotting
        ps = self.problem.pareto_set(use_cache=False, flatten=False)
        pf = self.problem.pareto_front(use_cache=False, flatten=False)

        # Design Space
        plot = Scatter(title="Design Space", axis_labels="x")
        plot.add(self.result.X, s=30, facecolors='none', edgecolors='r')
        if ps is not None:
            plot.add(ps, plot_type="line", color="black", alpha=0.7)
            plot.do()
            # plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
            # plot.apply(lambda ax: ax.set_ylim(-2, 2))
            plot.save(os.path.join(os.getcwd(), 'plots/design_space.png'))

        # Objective Space
        plot = Scatter(title="Objective Space")
        plot.add(self.result.F)
        if pf is not None:
            plot.add(pf, plot_type="line", color="black", alpha=0.7)
            plot.save(os.path.join(os.getcwd(), 'plots/objective_space.png'))

    def visualize(self):
        return Scatter().add(self.result.F)

    def viz_network(self):
        pass

    def _get_time_window(self, customer=None, bound='start'):
        if bound == 'start':
            bound = 'StartTime'
        elif bound == 'end':
            bound = 'EndTime'
        else:
            raise ValueError('Bound must be either start or end.')
        if customer is not None:
            return self.time_windows.loc[self.demand.loc[customer]['TimeWindow']][bound]

    def eval_timewindow(self, X):
        idx = np.where(X != 0)[0]
        veh_routes = np.split(X[idx], np.where(np.diff(idx) != 1)[0] + 1)
        df_dict = {}

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

                # Find time window violation
                if i != 0:
                    pickup_time = pickup_time + timedelta(minutes=travel_time)
                    if pickup_time < self._get_time_window(customer, bound='start'):
                        pickup_time = self._get_time_window(customer, bound='start')

                depart_time = pickup_time - timedelta(minutes=travel_time)

                df_dict[customer] = (pickup_time, veh_i, depart_time)

        df = pd.DataFrame.from_dict(df_dict, orient='index', columns=['ActualPickup', 'Vehicle', 'Departure'])
        df['CustomerID'] = df.index
        return df

    def plot_tw_violation(self, X):
        df = self.eval_timewindow(X)
        # df = copy.deepcopy()
        df = df.merge(self.demand, on='CustomerID')
        df['DesiredPickupStart'] = [self._get_time_window(c, bound='start') for c in df['CustomerID']]
        df['DesiredPickupEnd'] = [self._get_time_window(c, bound='end') for c in df['CustomerID']]
        df.drop(['OriginNodeID', 'DestinationNodeID', 'TimeWindow'], axis=1, inplace=True)

        df['Window'] = [(s, e) for s, e in zip(df['DesiredPickupStart'], df['DesiredPickupEnd'])]
        df['Violation'] = [(e, act) if act > e else None for e, act in zip(df['DesiredPickupEnd'], df['ActualPickup'])]
        colors = {v: np.random.rand(3,) for v in df['Vehicle']}

        df = df.sort_values(['Vehicle', 'ActualPickup'])
        df.reset_index(inplace=True)

        plt.figure(figsize=(12, 4), dpi=110)
        prev_veh, prev_pickup = -99, -99
        for i, customer in df.iterrows():
            if prev_veh != customer['Vehicle']:
                prev_pickup = min(df['Departure'])
            plt.plot([prev_pickup, customer['Departure'], customer['ActualPickup']], [i - 1, i - 1, i], linestyle='solid', color='grey', linewidth=0.5)
            plt.scatter(customer['Departure'], i - 1, color='grey', s=1)
            if customer['Violation'] is not None:
                plt.plot(customer['Violation'], [i, i], linestyle='dashed', color='red', linewidth=1)

            prev_veh = customer['Vehicle']
            prev_pickup = customer['ActualPickup']
        for i, customer in df.iterrows():
            plt.scatter(customer['ActualPickup'], i, color=colors[customer['Vehicle']], s=16)
            plt.plot(customer['Window'], [i, i], color=colors[customer['Vehicle']], linewidth=1)
        plt.savefig('plots/demand.jpg')

    @staticmethod
    def _od_to_dict(df):
        od_dict = {}
        for origin in df.index:
            od_dict[origin] = {}
            for dest in df.columns:
                od_dict[origin][dest] = df.loc[origin, dest]
        return od_dict

    @staticmethod
    def _get_bus_assignments(bus_schedule):
        bus_numbers = np.unique([c.split(' - ')[0] for c in bus_schedule.columns])
        bus_assignmnet = {bus_num: pd.unique(bus_schedule[f'{bus_num} - Nodes']) for bus_num in bus_numbers}
        return bus_assignmnet


if __name__ == '__main__':
    algo = Algorithm()
    algo.go(ngen=100, pop_size=50, n_offspring=20)
    algo.plot_tw_violation(algo.result.X[4])
