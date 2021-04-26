import glob

from pymoo.factory import get_termination
from pymoo.performance_indicator.hv import Hypervolume

import pandas as pd
import json
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTerminationWithRenormalization

from model_2e.crossover import *
from model_2e.mutation import *
from model_2e.sampling import *
from model_2e.problem import *
from model_2e.selection import *
from algorithm import Algorithm
from utils import *

import matplotlib.pyplot as plt
from matplotlib import cm
import time

plt.ioff()
plt.style.use('seaborn')


class BasicGA(Algorithm):
    def __init__(self, pop_size, n_offspring, problem, sampling, crossover, mutation, selection, termination):
        super().__init__(pop_size, n_offspring, problem, sampling, crossover, mutation, selection, termination)
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

    def get_convergence(self, save_data=None):
        if self.history is None:
            raise ValueError('Set save_history to True!')

        # ref_point = [10**np.ceil(np.log10(max(max(p), 1))) for p in list(map(list, zip(*self.history['F'])))]
        ref_point = [10000.0, 1000.0, 10000.0, 10000.0, 10.0, 1.0, 10000.0]
        # ref_point = np.array([10000, 2000, 400])
        # ref_point = np.array([10000, 2000, 400, 400, 10])

        # create the performance indicator object with reference point
        metric = Hypervolume(ref_point=ref_point, normalize=False)

        # calculate for each generation the HV metric
        hv = [metric.calc(f) for f in self.history['F']]

        num_gens = len(self.result.history)
        num_gen_to_feasible = sum(np.array([len(o.opt) for o in self.result.history]) == 1) - 1
        n_evals_to_feasible = self.history['n_evals'][num_gen_to_feasible]

        if save_data is not None:
            if algo.result.F is not None:
                if int(min([f[2] for f in algo.result.F])) == 0:
                    satisfies_all = True
                else:
                    satisfies_all = False
            else:
                satisfies_all = False

            data = {'n_evals': self.history['n_evals'],
                    'hypervolume': hv,
                    'satisfies_all': satisfies_all,
                    'num_gen_to_feasible': int(num_gen_to_feasible),
                    'n_evals_to_feasible': int(n_evals_to_feasible),
                    'num_gens': num_gens}

            with open(save_data, 'w') as outfile:
                json.dump(data, outfile)

        else:
            # visualze the convergence curve
            fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
            ax.plot(self.history['n_evals'], hv, '-o', markersize=4, linewidth=2)
            ax.set_title("Convergence")
            ax.set_xlabel("Function Evaluations")
            ax.set_ylabel("Hypervolume")
            fig.savefig(os.path.join(self.plot_dir, 'convergence.jpg'))
            plt.close(fig)


if __name__ == '__main__':
    print('Begin')
    # print('COMPARE POPULATION SIZES ---------------------------------------------------------------------------------------')
    # # for pop in [50, 100, 200, 400, 800]:
    # for pop in np.linspace(50, 400, 10):
    #     pop = int(pop)
    #     # termination = get_termination("n_gen", 400)
    #     termination = get_termination("n_eval", 50000)
    #     # termination = DesignSpaceToleranceTermination(tol=0.1, n_last=20, n_max_gen=1000)
    #     # termination = MultiObjectiveSpaceToleranceTermination(tol=0.0025, n_last=30, nth_gen=5, n_max_gen=1000, n_max_evals=None)
    #     # termination = MultiObjectiveSpaceToleranceTerminationWithRenormalization(tol=0.0025, n_last=30, nth_gen=5, n_max_gen=1000, n_max_evals=None)
    #
    #     algo = BasicGA(pop_size=pop,
    #                    n_offspring=pop,  # Default (None) uses the population size
    #                    problem=MyProblem,
    #                    sampling=BetterSample,
    #                    crossover=HybridCross,
    #                    mutation=RandomMutation,
    #                    selection=None,
    #                    termination=termination
    #                    )
    #
    #     algo.run(reduce_population_size_to=int(pop/3))
    #     algo.get_convergence(f'model_2e/results/pop_{pop}.json')
    #
    # # Save results
    # jsons = sorted(glob.glob('model_2e/results/population_1/pop*.json'))
    # virid = cm.get_cmap('viridis', 100)
    # dfs = {}
    # for j in jsons:
    #     dfs[os.path.basename(j).split('.')[0]] = pd.read_json(j)
    #
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    # for df in dfs:
    #     pop = int(df.split('_')[-1])
    #     ax.plot(dfs[df]['n_evals'], dfs[df]['hypervolume'], linewidth=1.5, label=pop, color=virid(pop/800))
    #
    # # ax.set_title("Convergence")
    # ax.set_xlabel("Function Evaluations")
    # ax.set_ylabel("Hypervolume")
    # ax.legend()
    # fig.savefig(os.path.join('model_2e/plots/compare_pop.jpg'))
    # plt.close(fig)

    # print('COMPARE MUTATION RATE -------------------------------------------------------------------------------------')
    # n_eval = 50000
    # for rate in np.linspace(0.06, 0.5, 20):
    #     start = time.time()
    #     termination = get_termination("n_eval", n_eval)
    #
    #     algo = BasicGA(pop_size=200,
    #                    n_offspring=200,  # Default (None) uses the population size
    #                    problem=MyProblem,
    #                    sampling=BetterSample,
    #                    crossover=HybridCross,
    #                    mutation=partialclass(RandomMutation, rate=rate),
    #                    selection=None,
    #                    termination=termination
    #                    )
    #
    #     algo.run(reduce_population_size_to=100)
    #     algo.get_convergence(f'model_2e/results/mutation_rate_{str(rate).replace(".", "")}.json')
    #     print(f'Ran {n_eval} in {(time.time() - start)/60} minutes')
    #
    # # Save results
    # virid = cm.get_cmap('viridis', 100)
    # jsons = sorted(glob.glob('model_2e/results/mutation_rate_*.json'))
    # dfs = {}
    # for j in jsons:
    #     dfs[os.path.basename(j).split('.')[0]] = pd.read_json(j)
    #
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    # for df in dfs:
    #     rate = round(float('0.' + df.split('_')[-1][1:]), 3)
    #     ax.plot(dfs[df]['n_evals'], dfs[df]['hypervolume'], linewidth=1.5, color=virid(rate/0.5))
    #
    # ax.set_xlabel("Function Evaluations")
    # ax.set_ylabel("Hypervolume")
    # # ax.legend()
    # fig.savefig(os.path.join('model_2e/plots/compare_mutation.jpg'))
    # plt.close(fig)
    #
    # fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    # for df in dfs:
    #     rate = float('0.' + df.split('_')[-1][1:])
    #     ax.scatter(rate, dfs[df]['n_evals_to_feasible'][0], label=rate, color='firebrick', s=14)
    #
    # # ax.set_title("Function Evaluations to Feasible Solution")
    # ax.set_xlabel("Mutation Rate")
    # ax.set_ylabel("Function Evaluations to Feasible Solution")
    # fig.savefig(os.path.join('model_2e/plots/func_evals_to_rate.jpg'))
    # plt.close(fig)



    # print('COMPARE CROSSOVER -----------------------------------------------------------------------------------------')
    # random.seed(42)
    # co_operators = ['cross_routes', 'cross_within_routes', 'cross_bus', 'cross_ss']
    # for weights in [[3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 1, 3], [1, 1, 1, 1], [1, 0, 1, 1]]:
    #     termination = get_termination("n_eval", 50000)
    #
    #     algo = BasicGA(pop_size=200,
    #                    n_offspring=200,  # Default (None) uses the population size
    #                    problem=MyProblem,
    #                    sampling=BetterSample,
    #                    crossover=functools.partial(HybridCross, weights=weights),
    #                    mutation=RandomMutation,
    #                    selection=None,
    #                    termination=termination
    #                    )
    #
    #     algo.run(reduce_population_size_to=100)
    #     algo.get_convergence(f'model_2e/results/crossover_2/crossover_{"".join(str(n) for n in weights)}.json')
    #
    # # Save results (dominators)
    # jsons = glob.glob('model_2e/results/crossover_2/crossover_*3*.json')
    # dfs = {}
    # for j in jsons:
    #     dfs[os.path.basename(j).split('.')[0]] = pd.read_json(j)
    #
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    # for df in dfs:
    #     label = co_operators[np.where(np.array(list(df[-4:])) == '3')[0][0]]
    #     ax.plot(dfs[df]['n_evals'], dfs[df]['hypervolume'], linewidth=2, label=label)
    #
    # ax.set_xlabel("Function Evaluations")
    # ax.set_ylabel("Hypervolume")
    # ax.legend()
    # fig.savefig(os.path.join('model_2e/plots/compare_crossover.jpg'))
    # plt.close(fig)
    #
    # # Save results (exclusions)
    # jsons = glob.glob('model_2e/results/crossover_1/crossover_*3*.json')
    # dfs = {}
    # for j in jsons:
    #     dfs[os.path.basename(j).split('.')[0]] = pd.read_json(j)
    #
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    # for df in dfs:
    #     label = co_operators[np.where(np.array(list(df[-4:])) == '3')[0][0]]
    #     ax.plot(dfs[df]['n_evals'], dfs[df]['hypervolume'], linewidth=1.5, color='silver')
    #
    # jsons = ['model_2e/results/crossover_1/crossover_1111.json', 'model_2e/results/crossover_1/crossover_1011.json']
    # labels = ['Uniform', 'Exclude cross_within_routes']
    # dfs = {}
    # for j in jsons:
    #     dfs[os.path.basename(j).split('.')[0]] = pd.read_json(j)
    #
    # for df, label in zip(dfs, labels):
    #     ax.plot(dfs[df]['n_evals'], dfs[df]['hypervolume'], linewidth=2, label=label)
    #
    # ax.set_xlabel("Function Evaluations")
    # ax.set_ylabel("Hypervolume")
    # ax.legend()
    # fig.savefig(os.path.join('model_2e/plots/compare_crossover_exclusions.jpg'))
    # plt.close(fig)


    # print('COMPARE RANDOM --------------------------------------------------------------------------------------------')
    # random.seed(42)
    # for _ in range(4):
    #     termination = get_termination("n_eval", 50000)
    #
    #     algo = BasicGA(pop_size=200,
    #                    n_offspring=200,  # Default (None) uses the population size
    #                    problem=MyProblem,
    #                    sampling=BetterSample,
    #                    crossover=Random,
    #                    mutation=Blank,
    #                    selection=None,
    #                    termination=termination
    #                    )
    #
    #     algo.run(reduce_population_size_to=100)
    #     algo.get_convergence(f'model_2e/results/blank{_}.json')
    #
    # # Save results (dominators)
    # jsons = glob.glob('model_2e/results/blank*.json')
    # dfs = {}
    # for j in jsons:
    #     dfs[os.path.basename(j).split('.')[0]] = pd.read_json(j)
    #
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    # for df in dfs:
    #     ax.plot(dfs[df]['n_evals'], dfs[df]['hypervolume'], linewidth=2)
    #
    # ax.set_xlabel("Function Evaluations")
    # ax.set_ylabel("Hypervolume")
    # ax.legend()
    # fig.savefig(os.path.join('model_2e/plots/random_walk.jpg'))
    # plt.close(fig)







    # if int(min([f[2] for f in algo.result.F])) == 0:
    #     print('Found solution satisfying all customers')
    #     for f in algo.result.F:
    #         if int(f[2]) == 0:
    #             print(f.astype(int))
    #
    # else:
    #     print(f'Did not find solution satisfying all customers. Best solution violates time windows by '
    #           f'a total of {int(min([f[2] for f in algo.result.F]))} minutes '
    #           f'(avg = {int(min([f[2] for f in algo.result.F]))/len(algo.data["demand"]["OriginNodeID"])} minutes)')

    # Find solutions with bus riders, if any.
    # for x in algo.result.X:
    #     if len(set(x[0][0][np.logical_or(x[0][1] > 0, x[0][2] > 0)])) > 0:
    #         print(x)

    # compare = False
    # if compare:
    #     algos, names = [], []
    #     samplings = {'S(SS)': SmartSortSample, 'S(R)': RandomSample}
    #     crossovers = {'C(H)': HybridCross, 'C(R)': CrossRoutes, 'C(A)': CrossAssignment}
    #     mutations = {'M(R)': RandomMutation}
    #     for sampling in samplings:
    #         for crossover in crossovers:
    #             for mutation in mutations:
    #                 algo = BasicGA(pop_size=50,
    #                                n_offspring=20,
    #                                problem=MyProblem,
    #                                sampling=samplings[sampling],
    #                                crossover=crossovers[crossover],
    #                                mutation=mutations[mutation],
    #                                # selection=TournamentSelection,
    #                                )
    #                 algo.run()
    #                 algos.append(algo)
    #                 names.append('/'.join([sampling, crossover, mutation]))
    #
    #     f0 = np.max([np.max([f[0] for f in algo.history['F']]) for algo in algos])
    #     f1 = np.max([np.max([f[1] for f in algo.history['F']]) for algo in algos])
    #     f2 = np.max([np.max([f[2] for f in algo.history['F']]) for algo in algos])
    #     f3 = np.max([np.max([f[3] for f in algo.history['F']]) for algo in algos])
    #     f4 = np.max([np.max([f[4] for f in algo.history['F']]) for algo in algos])
    #
    #     fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    #
    #     for algo, name_ in zip(algos, names):
    #         if algo.history is None:
    #             raise ValueError('Set save_history to True!')
    #
    #         ref_point = np.array([1.1, 1.1, 1.1, 1.1, 1.1])  # np.array([10000, 2000, 400, 400, 10])
    #
    #         # create the performance indicator object with reference point
    #         metric = Hypervolume(ref_point=ref_point, normalize=False)
    #
    #         norm_history = [np.array([f[0] / f0, f[1] / f1, f[2] / f2, f[3] / f3, f[4] / f4]) for f in
    #                         algo.history['F']]
    #         hv = [metric.calc(f) for f in norm_history]
    #
    #         # visualze the convergence curve
    #         ax.plot(algo.history['n_evals'], hv, '-o', markersize=4, linewidth=2, label=name_)
    #
    #     ax.set_title("Convergence")
    #     ax.set_xlabel("Function Evaluations")
    #     ax.set_ylabel("Hypervolume")
    #     ax.legend()
    #     fig.savefig('model_2e/plots/convergence.jpg')
    #     plt.close(fig)
    # else: