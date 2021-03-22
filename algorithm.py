from pymoo.algorithms.nsga2 import NSGA2
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination

from utils import *

import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()
plt.style.use('seaborn')


class Algorithm:
    def __init__(self, pop_size, n_offspring, problem, sampling, crossover, mutation, selection):
        xls = pd.ExcelFile(r'data/FictitiousDataMed.xlsx')
        self.demand = pd.read_excel(xls, 'CustomerDemand', header=0, index_col='CustomerID')
        self.demand.index = [c + 400 for c in self.demand.index]
        self.service_stations = pd.read_excel(xls, 'ServiceStations', header=0, index_col='ID')
        self.service_stations.index = [s + 500 for s in self.service_stations.index]

        bus_schedule = pd.read_excel(xls, 'PublicTransit', header=0)  # Bus schedule
        self.bus_assignment = self._get_bus_assignments(bus_schedule)  # Nodes visited by each bus
        self.bus_schedule = self._get_bus_schedule(bus_schedule)
        self.bus_stops = np.concatenate(list(self.bus_assignment.values()))

        time_windows = pd.read_excel(xls, 'TimeWindows', header=0, index_col='Code')
        time_windows['StartTime'] = [datetime.combine(date.today(), t) for t in time_windows['StartTime']]
        time_windows['EndTime'] = [datetime.combine(date.today(), t) for t in time_windows['EndTime']]
        self.time_windows = time_windows

        travel_distance = pd.read_excel(xls, 'OD-TravelDist', index_col=0, header=0)
        travel_time = pd.read_excel(xls, 'OD-TravelTime', index_col=0, header=0)
        self.travel_distance = self._od_to_dict(travel_distance)
        self.travel_time = self._od_to_dict(travel_time)

        self.n_vehicles = len(self.service_stations)

        self.data = {
            'travel_time': self.travel_time,
            'travel_distance': self.travel_distance,
            'service_stations': self.service_stations,
            'demand': self.demand,
            'time_windows': self.time_windows,
            'n_vehicles': self.n_vehicles,
            'bus_assignment': self.bus_assignment,
            'bus_stops': self.bus_stops,
            'bus_schedule': self.bus_schedule,
        }

        self.result = None
        self.history = None
        self.algorithm = None

        # termination = get_termination("n_gen", ngen)
        termination = DesignSpaceToleranceTermination(tol=0.1, n_last=20, n_max_gen=200)

        kwargs = {'pop_size': pop_size,
                  'n_offsprings': n_offspring,
                  'sampling': sampling(data=self.data),
                  'crossover': crossover(data=self.data),
                  'mutation': mutation(data=self.data),
                  'selection': selection,
                  'eliminate_duplicates': False, }
        kwargs = {k: kwargs[k] for k in kwargs if kwargs[k] is not None}

        algorithm = NSGA2(**kwargs)

        algorithm.setup(problem(data=self.data), termination=termination, seed=1, save_history=True, pf=True)
        self.algorithm = algorithm

    def run(self, save_history=True):
        if self.algorithm is None:
            raise ValueError('Do not use the Algorithm class directly.')
        # until the termination criterion has not been met
        self.history = {'F': [], 'n_evals': []}
        while self.algorithm.has_next():
            self.algorithm.next()  # Performs an iteration of the algorithm
            if save_history:
                self.history['F'].append(self.algorithm.opt.get('F').min(axis=0))
                self.history['n_evals'].append(self.algorithm.evaluator.n_eval)
            print(
                f'G.{self.algorithm.n_gen} - Best: {self.algorithm.opt.get("F").min(axis=0).astype(int)} - Number of NDS: {len(self.algorithm.opt)}')

        self.result = self.algorithm.result()  # Final results

    def _get_time_window(self, customer=None, bound='start'):
        if bound == 'start':
            bound = 'StartTime'
        elif bound == 'end':
            bound = 'EndTime'
        else:
            raise ValueError('Bound must be either start or end.')
        if customer is not None:
            return self.time_windows.loc[self.demand.loc[customer]['TimeWindow']][bound]

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
        bus_assignment = {bus_num: pd.unique(bus_schedule[f'{bus_num} - Nodes']) for bus_num in bus_numbers}
        return bus_assignment

    @staticmethod
    def _get_bus_schedule(bus_schedule):
        out = {}
        bus_numbers = np.unique([c.split(' - ')[0] for c in bus_schedule.columns])
        for bus_number in bus_numbers:
            out[bus_number] = {}

            times = bus_schedule[' - '.join([bus_number, 'Time'])]
            out[bus_number]['Time'] = np.array([datetime.combine(date.today(), t) for t in times])
            out[bus_number]['Nodes'] = np.array(bus_schedule[' - '.join([bus_number, 'Nodes'])].to_list())

        return out
