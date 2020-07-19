import itertools
import time
import numpy as np
import geopy.distance


file = "msg daten.csv"

def get_distances(list_entry1, list_entry2):
    coordinate1 = (list_entry1[1], list_entry1[2])
    coordinate2 = (list_entry2[1], list_entry2[2])

    # Distance
    return geopy.distance.distance(coordinate1, coordinate2).km


class heldkarp():
    def __init__(self, file):
        list_locations = []
        with open(file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:  # header
                    continue
                name_y_x = []
                name = line.split(",")[1]
                y = float(line.split(",")[6])
                x = float(line.split(",")[7])
                name_y_x.append(name)
                name_y_x.append(y)
                name_y_x.append(x)
                list_locations.append(name_y_x)
        distance_matrix = np.zeros((len(list_locations), len(list_locations)))
        for i, row_i in enumerate(list_locations):
            for j, row_j in enumerate(list_locations):
                if i == j:
                    continue
                else:
                    distance_matrix[i, j] = get_distances(row_i, row_j)

        self.distance_matrix = distance_matrix
        self.list_locations = list_locations

    def heldkarp(self, dist):

        print("SOLVING")
        n = len(dist)

        # Maps each subset of the nodes to the cost to reach that subset, as well
        # as what node it passed before reaching this subset.
        # Node subsets are represented as set bits.
        C = {}

        # Set transition cost from initial state
        for k in range(1, n):
            C[(1 << k, k)] = (dist[0][k], 0) # 1 << k means 1 * 2**k

        subset_time = time.time()
        # Iterate subsets of increasing length and store intermediate results
        # in classic dynamic programming manner
        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):
                # Set bits for all nodes in this subset
                bits = 0
                for bit in subset:
                    bits |= 1 << bit

                # Find the lowest cost to get to this subset
                for k in subset:
                    prev = bits & ~(1 << k)

                    result = []
                    for m in subset:
                        if m == 0 or m == k:
                            continue
                        result.append((C[(prev, m)][0] + dist[m][k], m))
                    C[(bits, k)] = min(result)
            print("Subset done: ", subset_size)
        print("All Subsets done")

        # We're interested in all bits but the least significant (the start state)
        bits = (2 ** n - 1) - 1

        # Calculate optimal cost
        res = []
        for k in range(1, n):
            res.append((C[(bits, k)][0] + dist[k][0], k))
        opt, parent = min(res)
        print("Optimal cost done")

        # Backtrack to find full path
        full_path = []
        for i in range(n - 1):
            full_path.append(parent)
            new_bits = bits & ~(1 << parent)
            _, parent = C[(bits, parent)]
            bits = new_bits
        print("Backtrack done")

        # Add implicit end state
        full_path.append(0)

        # Add implicit start state
        full_path.insert(0, 0)

        print("Subset Time: ", time.time() - subset_time)

        return opt, full_path

od = heldkarp(file)
cost, path = od.heldkarp(od.distance_matrix)

print("Cost = ", cost)
print("Path = ", path)



