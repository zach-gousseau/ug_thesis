from pymoo.performance_indicator.hv import Hypervolume

import pandas as pd

from model_2e.crossover import *
from model_2e.mutation import *
from model_2e.sampling import *
from model_2e.problem import *
from model_2e.selection import *
from algorithm import Algorithm
from utils import *

import matplotlib.pyplot as plt

plt.ioff()
plt.style.use('seaborn')


class BasicGA(Algorithm):
    def __init__(self, pop_size, n_offspring, problem, sampling, crossover, mutation, selection):
        super().__init__(pop_size, n_offspring, problem, sampling, crossover, mutation, selection)
        self.plot_dir = 'model_2e/plots/'
        make_dir(file_path=self.plot_dir)

    def eval_timewindow(self, X):
        X = X[0]
        veh_routes = split_at_delimiter(X)
        df_dict = {}
        bus_riders = {}  # customer: bus_number

        # Loop over each route
        for veh_i, veh_route in enumerate(veh_routes):
            # Initial pick-up time assumes picking up first customer on time (at their time window start)
            pickup_time = self._get_time_window(customer=veh_route[0], bound='start')
            ss_node = self.data['service_stations'].iloc[veh_i].NodeID

            i = 0
            while i < len(veh_route):

                customer = veh_route[i]  # Customer being serviced
                travel_time = 0  # Travel time of the one route

                if veh_route[i + 2] in self.bus_stops:
                    # Customer is dropped off at a bus stop
                    customer_org = self.data['demand'].loc[customer]['OriginNodeID']
                    customer_dest = veh_route[i + 2]
                    for bus in self.bus_assignment:
                        if veh_route[i + 2] in self.bus_assignment[bus]:
                            bus_riders[customer] = bus
                            break
                elif veh_route[i + 1] in self.bus_stops:
                    # Customer is picked up from a bus stop
                    customer_org = veh_route[i + 1]
                    customer_dest = self.data['demand'].loc[customer]['DestinationNodeID']

                else:
                    # Customer is only served by EAV
                    customer_org = self.data['demand'].loc[customer]['OriginNodeID']
                    customer_dest = self.data['demand'].loc[customer]['DestinationNodeID']

                # Add travel time and distance to pick up customer and drop them off
                travel_time += (
                            self.data['travel_time'][ss_node][customer_org] + self.data['travel_time'][customer_org][
                        customer_dest])

                ss_node = self.data['service_stations'].loc[veh_route[i + 1]].NodeID

                # Add travel time and distance from customer destination to service station destination
                travel_time += self.data['travel_time'][customer_dest][ss_node]

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

    def plot_all_timespace(self):
        for i in range(len(self.result.X)):
            self.plot_timespace(idx=i, fn=os.path.join(self.plot_dir, f'timespace/s{i}.jpg'))

    def plot_timespace(self, X=None, idx=None, fn=''):
        if idx is not None:
            X = self.result.X[idx]
        df = self.eval_timewindow(X)
        df = df.merge(self.demand, on='CustomerID')
        df['DesiredPickupStart'] = [self._get_time_window(c, bound='start') for c in df['CustomerID']]
        df['DesiredPickupEnd'] = [self._get_time_window(c, bound='end') for c in df['CustomerID']]
        df.drop(['OriginNodeID', 'DestinationNodeID', 'TimeWindow'], axis=1, inplace=True)

        df['Window'] = [(s, e) for s, e in zip(df['DesiredPickupStart'], df['DesiredPickupEnd'])]
        df['Violation'] = [(e, act) if act > e else None for e, act in zip(df['DesiredPickupEnd'], df['ActualPickup'])]
        colors = {v: np.random.rand(3, ) for v in df['Vehicle']}

        df = df.sort_values(['Vehicle', 'ActualPickup'])
        df.reset_index(inplace=True)

        fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
        prev_veh, prev_pickup = -99, -99
        for i, customer in df.iterrows():
            if prev_veh != customer['Vehicle']:
                prev_pickup = min(df['Departure'])
            ax.plot([prev_pickup, customer['Departure'], customer['ActualPickup']], [i - 1, i - 1, i],
                    linestyle='solid', color='grey', linewidth=0.5)
            ax.scatter(customer['Departure'], i - 1, color='grey', s=1)
            if customer['Violation'] is not None:
                ax.plot(customer['Violation'], [i, i], linestyle='dashed', color='red', linewidth=1)

            prev_veh = customer['Vehicle']
            prev_pickup = customer['ActualPickup']
        for i, customer in df.iterrows():
            ax.scatter(customer['ActualPickup'], i, color=colors[customer['Vehicle']], s=16)
            ax.plot(customer['Window'], [i, i], color=colors[customer['Vehicle']], linewidth=1)
        if idx is None:
            ax.set_title('Solution')
        else:
            f = self.result.F[idx]
            ax.set_title(f'Solution -- f1={f[0]} | f2={f[1]} | f3={f[2]} | f4={f[3]} | f5={f[4]}')
        ax.set_xlabel('Date/Time')
        ax.set_ylabel('Distance (NOT TO SCALE)')
        if fn == '':
            fig.savefig(os.path.join(self.plot_dir, 'demand.jpg'))
        else:
            make_dir(file_path=fn)
            fig.savefig(fn)
        plt.close(fig)

    def plot_convergence(self):
        if self.history is None:
            raise ValueError('Set save_history to True!')
        fig, ax = plt.subplots(figsize=(8, 6), dpi=110)

        ref_point = np.array([10000, 2000, 400])
        # ref_point = np.array([10000, 2000, 400, 400, 10])

        # create the performance indicator object with reference point
        metric = Hypervolume(ref_point=ref_point, normalize=False)

        # calculate for each generation the HV metric
        hv = [metric.calc(f) for f in self.history['F']]

        # visualze the convergence curve
        ax.plot(self.history['n_evals'], hv, '-o', markersize=4, linewidth=2)
        ax.set_title("Convergence")
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Hypervolume")
        fig.savefig(os.path.join(self.plot_dir, 'convergence.jpg'))
        plt.close(fig)


if __name__ == '__main__':
    compare = False
    if compare:
        algos, names = [], []
        samplings = {'S(SS)': SmartSortSample, 'S(R)': RandomSample}
        crossovers = {'C(H)': HybridCross, 'C(R)': CrossRoutes, 'C(A)': CrossAssignment}
        mutations = {'M(R)': RandomMutation}
        for sampling in samplings:
            for crossover in crossovers:
                for mutation in mutations:
                    algo = BasicGA(pop_size=50,
                                   n_offspring=20,
                                   problem=MyProblem,
                                   sampling=samplings[sampling],
                                   crossover=crossovers[crossover],
                                   mutation=mutations[mutation],
                                   # selection=TournamentSelection,
                                   )
                    algo.run()
                    algos.append(algo)
                    names.append('/'.join([sampling, crossover, mutation]))

        f0 = np.max([np.max([f[0] for f in algo.history['F']]) for algo in algos])
        f1 = np.max([np.max([f[1] for f in algo.history['F']]) for algo in algos])
        f2 = np.max([np.max([f[2] for f in algo.history['F']]) for algo in algos])
        f3 = np.max([np.max([f[3] for f in algo.history['F']]) for algo in algos])
        f4 = np.max([np.max([f[4] for f in algo.history['F']]) for algo in algos])

        fig, ax = plt.subplots(figsize=(8, 6), dpi=110)

        for algo, name_ in zip(algos, names):
            if algo.history is None:
                raise ValueError('Set save_history to True!')

            ref_point = np.array([1.1, 1.1, 1.1, 1.1, 1.1])  # np.array([10000, 2000, 400, 400, 10])

            # create the performance indicator object with reference point
            metric = Hypervolume(ref_point=ref_point, normalize=False)

            norm_history = [np.array([f[0] / f0, f[1] / f1, f[2] / f2, f[3] / f3, f[4] / f4]) for f in
                            algo.history['F']]
            hv = [metric.calc(f) for f in norm_history]

            # visualze the convergence curve
            ax.plot(algo.history['n_evals'], hv, '-o', markersize=4, linewidth=2, label=name_)

        ax.set_title("Convergence")
        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel("Hypervolume")
        ax.legend()
        fig.savefig('model_2e/plots/convergence.jpg')
        plt.close(fig)
    else:
        algo = BasicGA(pop_size=300,
                       n_offspring=300,  # Default (None) uses the population size
                       problem=MyProblem,
                       sampling=BetterSample,
                       crossover=HybridCross,
                       mutation=RandomMutation,
                       selection=None,
                       )

        algo.run(reduce_population_size_to=100)

        if int(min([f[2] for f in algo.result.F])) == 0:
            print('Found solution satisfying all customers')
            for f in algo.result.F:
                if int(f[2]) == 0:
                    print(f)

        else:
            print(f'Did not find solution satisfying all customers. Best solution violates time windows by '
                  f'a total of {int(min([f[2] for f in algo.result.F]))} minutes')

        # algo.plot_timespace(idx=-2)
        # algo.plot_all_timespace()
        algo.plot_convergence()

        # Find solutions with bus riders, if any.
        for x in algo.result.X:
            if len(set(x[0][0][np.logical_or(x[0][1] > 0, x[0][2] > 0)])) > 0:
                print(x)
