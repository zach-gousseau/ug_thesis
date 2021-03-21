import random
from utils import *
from pymoo.model.mutation import Mutation


class RandomMutation(Mutation):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _do(self, problem, X, **kwargs):
        if random.random() < 0.0:
            # for each individual
            for i in range(len(X)):
                choice = random.randint(1, 3)
                if choice == 1:
                    # Switch two customers
                    X[i] = self.switch_customers(X[i])
                elif choice == 2:
                    # Randomly change ss
                    X[i] = self.change_service_station(X[i])
                elif choice == 3:
                    X[i] = self.change_bus_stop(X[i])

                # Checks
                customers_in_chromosome = X[i][np.logical_and(X[i] > 400, X[i] < 500)]
                for customer in self.data['demand'].index:
                    print(customer in customers_in_chromosome)
                    assert customer in customers_in_chromosome

                # for veh_route in split_at_delimiter(X[i]):
                #     assert (len(veh_route) % 2) == 0

        return X

    def switch_customers(self, X):
        c1 = random.choice(self.data['demand'].index)
        c2 = random.choice(self.data['demand'].index)
        while c1 == c2:
            c2 = random.choice(self.data['demand'].index)

        c1_idx = np.where(X == c1)[0][0]
        c2_idx = np.where(X == c2)[0][0]
        X[0][[c1_idx, c2_idx]] = X[0][[c2_idx, c1_idx]]
        return X

    def change_service_station(self, X):
        old_ss_idx = random.choice(np.where(np.in1d(X[0], self.data['service_stations'].index))[0])
        new_ss = random.choice(self.data['service_stations'].index)
        while X[0][old_ss_idx] == new_ss:
            new_ss = random.choice(self.data['service_stations'].index)
        X[0][old_ss_idx] = new_ss
        return X

    def change_bus_stop(self, X):
        old_ss_idx = random.choice(np.where(np.in1d(X[0], self.data['bus_stops'].index))[0])
        new_ss = random.choice(self.data['bus_stops'].index)
        while X[0][old_ss_idx] == new_ss:
            new_ss = random.choice(self.data['bus_stops'].index)
        X[0][old_ss_idx] = new_ss
        return X