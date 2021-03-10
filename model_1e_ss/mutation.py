import random
from utils import *
from pymoo.model.mutation import Mutation


class RandomMutation(Mutation):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _do(self, problem, X, **kwargs):
        # for each individual
        for i in range(len(X)):
            # Switch two customers
            c1 = random.choice(self.data['demand'].index)
            c2 = random.choice(self.data['demand'].index)
            while c1 == c2:
                c2 = random.choice(self.data['demand'].index)

            c1_idx = np.where(X[i] == c1)[0][0]
            c2_idx = np.where(X[i] == c2)[0][0]
            X[i][[c1_idx, c2_idx]] = X[i][[c2_idx, c1_idx]]
            for veh_route in split_at_delimiter(X[i]):
                assert (len(veh_route) % 2) == 0
        return X
