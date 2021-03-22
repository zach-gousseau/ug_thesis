import math

import numpy as np

from pymoo.model.selection import Selection
from pymoo.util.misc import random_permuations

from pymoo.operators.selection.tournament_selection import compare
from pymoo.util.dominator import Dominator


class TournamentSelection(Selection):
    """
      The Tournament selection is used to simulated a tournament between individuals. The pressure balances
      greedy the genetic algorithm will be.
    """

    def __init__(self, func_comp=None, pressure=2):
        """
        Parameters
        ----------
        func_comp: func
            The function to compare two individuals. It has the shape: comp(pop, indices) and returns the winner.
            If the function is None it is assumed the population is sorted by a criterium and only indices are compared.
        pressure: int
            The selection pressure to bie applied. Default it is a binary tournament.
        """

        # selection pressure to be applied
        self.pressure = pressure

        self.f_comp = func_comp
        if self.f_comp is None:
            raise Exception("Please provide the comparing function for the tournament selection!")

    def _do(self, pop, n_select, n_parents=1, **kwargs):
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop))[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.f_comp(pop, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))


def compare(a, a_val, b, b_val, method, return_random_if_equal=False):
    if method == 'larger_is_better':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None
    elif method == 'smaller_is_better':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None
    else:
        raise Exception("Unknown method.")


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].get("rank"), b, pop[b].get("rank"),
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)