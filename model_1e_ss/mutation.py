import random
from utils import *
from pymoo.model.mutation import Mutation


class RandomMutation(Mutation):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _do(self, problem, X, **kwargs):
        if random.random() < 0.15:
            # for each individual
            for i in range(len(X)):
                if random.random() > 0.5:

                    # Switch two customers
                    X[i] = self.switch_customers(X[i])
                else:
                    # Randomly change ss
                    X[i] = self.change_service_station(X[i])

                for veh_route in split_at_delimiter(X[i]):
                    assert (len(veh_route) % 2) == 0
        return X

    def switch_customers(self, X):
        c1 = random.choice(self.data['demand'].index)
        c2 = random.choice(self.data['demand'].index)
        while c1 == c2:
            c2 = random.choice(self.data['demand'].index)

        c1_idx = np.where(X == c1)[0][0]
        c2_idx = np.where(X == c2)[0][0]
        X[[c1_idx, c2_idx]] = X[[c2_idx, c1_idx]]
        return X

    def change_service_station(self, X):
        old_ss_idx = random.choice(np.where(np.in1d(X, self.data['service_stations'].index))[0])
        new_ss = random.choice(self.data['service_stations'].index)
        while X[old_ss_idx] == new_ss:
            new_ss = random.choice(self.data['service_stations'].index)
        X[old_ss_idx] = new_ss
        return X