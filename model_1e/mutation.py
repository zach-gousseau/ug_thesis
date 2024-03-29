import random

from pymoo.model.mutation import Mutation


class RandomMutation(Mutation):
    def __init__(self, **data):
        super().__init__()
        self.data = data

    def _do(self, problem, X, **kwargs):
        # for each individual
        for i in range(len(X)):
            idx_1, idx_2 = random.randint(0, len(X[i]) - 1), random.randint(0, len(X[i]) - 1)
            X[i][[idx_1, idx_2]] = X[i][[idx_2, idx_1]]
        return X
