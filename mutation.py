import random

from pymoo.model.mutation import Mutation


class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        # for each individual
        for i in range(len(X)):
            idx_1, idx_2 = random.randint(0, len(X[i]) - 1), random.randint(0, len(X[i]) - 1)
            X[i][[idx_1, idx_2]] = X[i][[idx_2, idx_1]]
        return X
