import copy
import os

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination

import numpy as np
import pandas as pd


class MyProblem(Problem):
    def __init__(self):
        xls = pd.ExcelFile(r'data/FictitiousData.xlsx')
        self.time_windows = pd.read_excel(xls, 'TimeWindows', header=0, index_col='Code')
        self.demand = pd.read_excel(xls, 'CustomerDemand', header=0)
        self.service_stations = pd.read_excel(xls, 'ServiceStations', header=0)
        self.bus_schedule = pd.read_excel(xls, 'PublicTransit', header=0)  # Bus schedule
        self.bus_assignment = self._get_bus_assignments(self.bus_schedule)  # Nodes visited by each bus

        travel_distance = pd.read_excel(xls, 'OD-TravelDist', index_col=0, header=0)
        travel_time = pd.read_excel(xls, 'OD-TravelTime', index_col=0, header=0)
        self.travel_distance = self._od_to_dict(travel_distance)
        self.travel_time = self._od_to_dict(travel_time)

        # locations = travel_time.index
        # n_genes = 4 * len(self.demand)

        #  [ss, c1, ss, c2, ss, c3, ss, c4, ..., cn, ss]

        super().__init__(n_var=1,  # Number of genes
                         n_obj=2,  # Number of objectives
                         # xl=np.array([min(locations)] * n_genes),  # Lower bound for genes (should be same length as value of n_var)
                         # xu=np.array([max(locations)] * n_genes),  # Upper bound (ditto)
                         type_var=np.int,
                         elementwise_evaluation=True,
                         )

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

    def _evaluate(self, X, out, *args, **kwargs):
        # Define the fitness functions - should have same number of fitness functions as n_obj)

        # Travel time
        f1 = np.zeros(len(X))
        for i in range(X.shape[1] - 1):
            f1 += np.array([self.travel_time[o][d] for o, d in zip(X[:, i], X[:, i+1])])

        # Travel distance
        f2 = np.zeros(len(X))
        for i in range(X.shape[1] - 1):
            f2 += np.array([self.travel_distance[o][d] for o, d in zip(X[:, i], X[:, i+1])])

        # Every second item should be a service station
        # g1 = np.all(np.isin(X[:, ::2], self.service_stations.NodeID), axis=1)

        out["F"] = np.column_stack([f1, f2])


class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros(shape=(n_samples, 1), dtype=np.int)

        for i in range(n_samples):
            X[i, 0] = "".join([np.random.choice(problem.ALPHABET) for _ in range(problem.n_characters)])

        return X

class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=np.object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]

            # prepare the offsprings
            off_a = ["_"] * problem.n_characters
            off_b = ["_"] * problem.n_characters

            for i in range(problem.n_characters):
                if np.random.random() < 0.5:
                    off_a[i] = a[i]
                    off_b[i] = b[i]
                else:
                    off_a[i] = b[i]
                    off_b[i] = a[i]

            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = "".join(off_a), "".join(off_b)

        return Y


class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):

        # for each individual
        for i in range(len(X)):

            r = np.random.random()

            # with a probabilty of 40% - change the order of characters
            if r < 0.4:
                perm = np.random.permutation(problem.n_characters)
                X[i, 0] = "".join(np.array([e for e in X[i, 0]])[perm])

            # also with a probabilty of 40% - change a character randomly
            elif r < 0.8:
                prob = 1 / problem.n_characters
                mut = [c if np.random.random() > prob
                       else np.random.choice(problem.ALPHABET) for c in X[i, 0]]
                X[i, 0] = "".join(mut)

        return X

if __name__ == '__main__':
    problem = MyProblem()
    termination = get_termination("n_gen", 40)
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_sbx", prob=0.9, eta=15),
        mutation=get_mutation("int_pm", eta=20),
        eliminate_duplicates=True
    )

    # Optimize a copy of the algorithm to ensure reproducibility
    obj = copy.deepcopy(algorithm)

    obj.setup(problem, termination=termination, seed=1)

    # until the termination criterion has not been met
    while obj.has_next():
        obj.next()  # Performs an iteration of the algorithm

        # Print current iteration results
        print(f"gen: {obj.n_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV').min()} ideal: {obj.opt.get('F').min(axis=0)}")

    result = obj.result()  # Final results

    # Get the pareto-set and pareto-front for plotting
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)

    # Design Space
    plot = Scatter(title="Design Space", axis_labels="x")
    plot.add(result.X, s=30, facecolors='none', edgecolors='r')
    if ps is not None:
        plot.add(ps, plot_type="line", color="black", alpha=0.7)
        plot.do()
        # plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
        # plot.apply(lambda ax: ax.set_ylim(-2, 2))
        plot.save(os.path.join(os.getcwd(), 'plots/design_space.png'))

    # Objective Space
    plot = Scatter(title="Objective Space")
    plot.add(result.F)
    if pf is not None:
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
        plot.save(os.path.join(os.getcwd(), 'plots/objective_space.png'))