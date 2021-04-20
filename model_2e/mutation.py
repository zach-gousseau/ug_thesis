import random
from utils import *
from pymoo.model.mutation import Mutation
import copy
from collections import Counter


class RandomMutation(Mutation):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if random.random() < 0.05:
                # for each individual
                choice = random.randint(1, 4)
                if choice == 1:
                    # Switch two customers
                    X[i] = self.switch_customers(X[i])
                elif choice == 2:
                    # Randomly change ss
                    X[i] = self.change_service_station(X[i])
                elif choice == 3:
                    X[i] = self.change_bus_stop(X[i])
                elif choice == 4:
                    X[i] = self.add_bus_rider(X[i])

            assert sum(X[i][0][1] > 0) == sum(X[i][0][2] > 0)
        return X

    def switch_customers(self, X):
        c1 = random.randint(0, len(X[0]))
        while X[0][0][c1] == 0:
            c1 = random.randint(0, len(X[0]))

        c2 = random.randint(0, len(X[0]))
        while X[0][0][c2] == 0 or c1 == c2:
            c2 = random.randint(0, len(X[0]))

        c1_entry, c2_entry = copy.deepcopy(X[0][:, c1]), copy.deepcopy(X[0][:, c2])
        X[0][:, c1], X[0][:, c2] = c2_entry, c1_entry
        return X

    def change_service_station(self, X):
        old_ss_idx = random.randint(0, len(X[0]))
        while X[0][0][old_ss_idx] == 0:
            old_ss_idx = random.randint(0, len(X[0]))

        new_ss = random.choice(self.data['service_stations'].index)
        while X[0][3][old_ss_idx] == new_ss:
            new_ss = random.choice(self.data['service_stations'].index)
        X[0][3][old_ss_idx] = new_ss
        return X

    def add_bus_rider(self, X):
        X0_new = copy.deepcopy(X[0])

        # customer = random.choice([item for item, count in Counter(X[0][0]).items() if (count == 1 and item != 0)])
        try:
            idx = random.choice(np.argwhere(np.logical_and(X0_new[1] == -1, X0_new[2] == -1)))
        except IndexError:
            return X
        customer = X0_new[0][idx]
        # idx = np.where(X[0][0] == customer)[0][0]
        on_bs = random.choice(self.data['bus_stops'])
        off_bs = random.choice(self.data['bus_stops'])
        while off_bs == on_bs:
            off_bs = random.choice(self.data['bus_stops'])

        X0_new[1][idx] = on_bs

        entry = [customer, -1, off_bs, random.choice(self.data['service_stations'].index)]

        insert_point = random.randint(0, len(X0_new))  # Choose random insert point
        X0_new = np.insert(X0_new, insert_point, entry, axis=1)

        bus_riders = list(set(X0_new[0][np.logical_or(X0_new[1] > 0, X0_new[2] > 0)]))
        for b in bus_riders:
            assert len(np.where(X0_new[0] == b)[0]) == 2

        assert sum(X0_new[1] > 0) == sum(X0_new[2] > 0)
        X[0] = X0_new
        return X

    def change_bus_stop(self, X):
        level = random.choice([1, 2])

        try:
            # Find one bus station to switch up
            old_bs_idx = random.choice(np.where(X[0][level] > 0)[0])

            new_bs = random.choice(self.data['bus_stops'])
            while X[0][level][old_bs_idx] == new_bs:
                new_bs = random.choice(self.data['bus_stops'])

            X[0][level][old_bs_idx] = new_bs

        except IndexError:
            return self.add_bus_rider(X)

        return X