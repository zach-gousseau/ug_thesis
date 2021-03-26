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
            if random.random() < 1:
                # for each individual
                choice = random.randint(0, 3)
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
        print(X)
        customer = random.choice([item for item, count in Counter(X[0][0]).items() if (count == 1 and item != 0)])
        idx = np.where(X[0][0] == customer)[0][0]
        bus_choice = random.choice(list(self.data['bus_assignment'].keys()))
        on_bs = random.choice(self.data['bus_assignment'][bus_choice])
        off_bs = random.choice(self.data['bus_assignment'][bus_choice])

        while off_bs == on_bs:
            off_bs = random.choice(self.data['bus_assignment'][bus_choice])

        X[0][1][idx] = on_bs

        entry = [customer, -1, off_bs, random.choice(self.data['service_stations'].index)]

        insert_point = random.randint(0, len(X[0]))  # Choose random insert point
        X[0] = np.insert(X[0], insert_point, entry, axis=1)
        print(X)
        if X[0][0][0] == 401 and X[0][0][1] == 402 and X[0][0][2] == 420:
            pass

        bus_riders = list(set(X[0][0][np.logical_or(X[0][1] > 0, X[0][2] > 0)]))
        for b in bus_riders:
            assert len(np.where(X[0][0] == b)[0]) == 2
        return X

    def change_bus_stop(self, X):
        level = random.choice([1, 2])

        try:
            # Find one bus station to switch up
            old_bs_idx = random.choice(np.where(X[0][level] > 0)[0])

            bus = self.data['bus_stop_assignment'][X[0][level][old_bs_idx]]
            new_bs = random.choice(self.data['bus_assignment'][bus])
            while X[0][level][old_bs_idx] == new_bs:
                new_bs = random.choice(self.data['bus_assignment'][bus])

            X[0][level][old_bs_idx] = new_bs

        except IndexError:
            return X
            return self.add_bus_rider(X)

        return X