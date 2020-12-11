# this is the way

from copy import copy, deepcopy
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *
import heapq
import itertools

class Stack(object):

    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def isEmpty(self):
        return len(self.items) == 0


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    # Greedy algorithm to get initial BSSF
    # space is O(n) - has to hold all of the cities
    #

    def greedy(self, time_allowance=60.0):
        bssf = None
        cities = self._scenario.getCities()
        solution_dict = {}
        start_time = time.time()
        # 60 seconds max time O(1)
        while time.time() - start_time < time_allowance:
            # O(n) - iterates through the size of all the cities
            for index in range(len(cities)):
                path = []
                city = cities[index]
                current_city = cities[index]
                path.append(city)
                to_visit = deepcopy(cities)
                del to_visit[index]
                # O(n) - iterates through size of inputs
                # takes one out every iteration
                while len(to_visit) != 0:
                    closest_city_tuple = self.get_closest_cities(current_city, to_visit)[0]
                    closest_city_index = to_visit.index(closest_city_tuple[0])
                    closest_city = to_visit[closest_city_index]
                    del to_visit[closest_city_index]
                    path.append(closest_city)
                    current_city = closest_city
                bssf = TSPSolution(path)
                end_time = time.time()
                results = {}
                results['cost'] = bssf.cost
                results['time'] = end_time - start_time
                results['count'] = None
                results['soln'] = bssf
                results['max'] = None
                results['total'] = None
                results['pruned'] = None
                solution_dict[index] = results

            self.lowest_cost = float("inf")
            for key, solution in solution_dict.items():
                print(key, solution["cost"])
                if solution["cost"] < self.lowest_cost:
                    self.lowest_cost = solution["cost"]
                    lowest = solution

            return lowest

    # O(n) time and space
    def get_closest_cities(self, city, city_list):
        cost = {}
        for city_to_visit in city_list:
            cost[city_to_visit] = city.costTo(city_to_visit)

        sorted_x = sorted(cost.items(), key=lambda kv: kv[1])
        return sorted_x

    #ZONK

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    # k = num states made ( and also checked)
    # TIME - O(k*n*n)
    # SPACE - O(k*n*n) (states on the heap)
    def branchAndBound(self, time_allowance=60.0):
            # SET UP - O(n) space and O(1) time
            heap = []
            bssf_changes = 0
            max_length = 1
            num_states_made = 1
            pruned = 0
            num_solutions = 0
            # set bssf to solution from greedy
            bssf = self.greedy(time_allowance=time_allowance)['soln']

            cities = self._scenario.getCities()
            self.cities = cities
            self.lowest_ave_cost = float("inf")
            self.lowest_cost = bssf.cost
            # O(n^2) time and space
            first_reduced_matrix, first_lb = self.get_init_reduced_matrix(cities)
            first_city = tuple((first_lb, cities[0], cities[1:], first_reduced_matrix, [cities[0]._index], first_lb))
            # 0 - val to use for sorting
            # 1 - current node
            # 2 - the rest of the nodes to still visit
            # 3 - reduced matrix for current node
            # 4 - path to current spot
            # 5 - total cost of current path (current node)
            heapq.heappush(heap, first_city)
            # BEGIN
            start_time = time.time()
            while time.time() - start_time < time_allowance and len(heap):
                # pop/push -> O(logn) must shift everything around but only O(1) space
                next_city = heapq.heappop(heap)
                # O(1) comparison
                if next_city[5] < self.lowest_cost:
                    for city in next_city[2]:
                        # there must be a path/edge
                        if self._scenario._edge_exists[next_city[1]._index][city._index]:
                            # O(n^2) time and space
                            new_expanded_problem = self.get_reduced_matrix(city, next_city[3], next_city)
                            # hit bottom, keep solution
                            if not len(new_expanded_problem[2]):
                                # O(n) to get the cities
                                route = self.convert_to_cities(new_expanded_problem[4])
                                bssf = TSPSolution(route)
                                if bssf.cost < self.lowest_cost:
                                    self.lowest_cost = min(bssf.cost, self.lowest_cost)
                                    bssf_changes += 1
                                num_solutions += 1
                            # still hasn't hit bottom
                            else:
                                # keep the node
                                if new_expanded_problem[5] < self.lowest_cost:
                                    # It is O(logn) time to heapify after inserting
                                    # O(1) space
                                    heapq.heappush(heap, new_expanded_problem)
                                    num_states_made += 1
                                # prune it - the cost is already higher than our lowest cost
                                else:
                                    pruned += 1
                                    num_states_made += 1
                else:
                    pruned += 1
                    num_states_made += 1
                # O(1) time and space
                max_length = self.getMax(len(heap), max_length)

            final_time = time.time()
            results = {}
            results['cost'] = self.lowest_cost
            results['max'] = max_length
            results['total'] = num_states_made
            results['pruned'] = pruned
            results['count'] = num_solutions
            results['soln'] = bssf
            results['time'] = final_time - start_time
            return results


    # O(n^2) time and space
    # gets distance from first city to all the other cities (that have edges)
    # similar to get_reduced_matrix, but just with the first part
    # see get_reduced_matrix method for more details about what is happening
    def get_init_reduced_matrix(self, city_list):
        reduced_matrix = np.full((len(city_list), len(city_list)), fill_value=np.inf)
        # O(n)
        for origin_index, city in enumerate(city_list):
            for dest_index, dest_city in enumerate(city_list):
                if origin_index == dest_index:
                    continue
                distance = city.costTo(dest_city)
                reduced_matrix[origin_index][dest_index] = distance

        sum = 0
        #O(n^2) to go through every cell in the matrix
        for row in range(reduced_matrix.shape[0]):
            row_min = np.min(reduced_matrix[row])
            reduced_matrix[row] = reduced_matrix[row] - row_min
            sum += row_min

        for col in range(reduced_matrix.shape[1]):
            column_min = np.min(reduced_matrix[:, col])
            reduced_matrix[:, col] = reduced_matrix[:, col] - column_min
            sum += column_min

        return reduced_matrix, sum


    # O(1) time and space - just compares the two values
    def getMax(self, x, y):
        return max(x,y)

    # O(1) time and space
    def get_value(self, value, cities_visited):
        size = len(cities_visited)
        if len(cities_visited):
            return value / size
        else:
            return 0

    # O(n) time and space
    # iterates through all of the cities
    def convert_to_cities(self, city_indices):
        # create empty list
        cities = []
        for num in city_indices:
            cities.append(self.cities[num])
        return cities

    # O(n) time, O(1) space
    # goes through cities, deletes by id
    def delete_city_by_id(self, cities_to_still_visit, next_city_to_visit):
        for index, city in enumerate(cities_to_still_visit):
            if city._index == next_city_to_visit._index:
                delete_index = index
                break

        del cities_to_still_visit[delete_index]
        return cities_to_still_visit

    # O(n^2) time and space
    # iterates through array of size (n^2) to change cells
    # this function does what we did for hw and in class, we first go through the rows and reduce them all,
    # then we go through the columns are reduce them all by the lowest value too
    # after going through the rows and the columns, we check to make sure that each row and column has a zero in it
    def get_reduced_matrix(self, next_city_to_visit, matrix, given_tuple):
        # copy to avoid changes to original
        # O(1) time, O(n^2) space
        city_copy = deepcopy(given_tuple)
        matrix = matrix.copy()
        sum = 0
        initial_cost = matrix[city_copy[1]._index][next_city_to_visit._index]

        # BEGIN REDUCING MATRIX
        matrix[city_copy[1]._index] = np.inf
        matrix[:, next_city_to_visit._index] = np.inf
        matrix[next_city_to_visit._index][city_copy[1]._index] = np.inf

        for row in range(matrix.shape[0]):
            row_min = np.min(matrix[row])
            if not np.isinf(row_min):
                matrix[row] = matrix[row] - row_min
                sum += row_min

        for col in range(matrix.shape[1]):
            column_min = np.min(matrix[:, col])
            if not np.isinf(column_min):
                matrix[:, col] = matrix[:, col] - column_min
                sum += column_min

        # O(n) time, O(1) space to delete it
        # visits all the cities
        places_to_go = city_copy[2]
        places_to_go = self.delete_city_by_id(places_to_go, next_city_to_visit)

        updated_cost = city_copy[5] + initial_cost + sum

        result = ((self.get_value(city_copy[5], city_copy[4]), next_city_to_visit, places_to_go, matrix, city_copy[4] + [next_city_to_visit._index], updated_cost))

        return result

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

####################################################################################################################
######################################################  FANCY  ###################################################
####################################################################################################################


    def fancy(self, time_allowance=60.0):
        self._fancy_count = 0
        print('fancy')
        results = {}
        start_time = time.time()

        greedy_route = self.greedy()['soln'].route
        sub_routes = []
        current_sub_route = []

        for i in range(len(greedy_route)):
            current_sub_route.append(greedy_route[i])
            greedy_route[i]._sub_route_index = i % 16

            if i % 16 == 0 and i != 0:
                if i == len(greedy_route) - 1: current_sub_route.append(greedy_route[0])
                sub_routes.append(current_sub_route.copy())
                current_sub_route = [greedy_route[i]]
            elif i == len(greedy_route) - 1:
                current_sub_route.append(greedy_route[0])
                sub_routes.append(current_sub_route)

        #bb_sub_routes = []
        total_route = []
        for route in sub_routes:
            # run B and B on each subroute
            #bb_sub_routes.append(self.branchAndBoundFancy(route)['soln'].route)
            total_route = total_route + self.branchAndBoundFancy(route)['soln'].route


        bssf = TSPSolution(total_route)

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = -1
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    def branchAndBoundFancy(self, fCities, time_allowance=60.0):
        heap = []
        bssf_changes = 0
        max_length = 1
        num_states_made = 1
        pruned = 0
        num_solutions = 0
        bssf = self.greedy(time_allowance=time_allowance)['soln']
        cities = fCities
        self.lowest_ave_cost = float("inf")
        self.lowest_cost = bssf.cost
        linking_city = cities.pop()  # = cities[:len(cities)-1]
        print(linking_city._name)
        self.cities = cities
        first_reduced_matrix, first_lb = self.get_fancy_init_reduced_matrix(cities, linking_city)
        first_city = tuple(
            (first_lb, cities[0], cities[1:], first_reduced_matrix, [cities[0]._sub_route_index], first_lb, 0))

        # 0 - val to use for sorting
        # 1 - current node
        # 2 - the rest of the nodes to still visit
        # 3 - reduced matrix for current node
        # 4 - path to current spot
        # 5 - total cost of current path (current node)
        # 6 - index within sub-route

        heapq.heappush(heap, first_city)
        start_time = time.time()
        while len(heap):
            next_city = heapq.heappop(heap)
            if next_city[5] < self.lowest_cost:
                for city in next_city[2]:
                    if self._scenario._edge_exists[next_city[1]._sub_route_index][city._sub_route_index]:
                        new_expanded_problem = self.get_fancy_reduced_matrix(city, next_city)
                        if not len(new_expanded_problem[2]):
                            route = self.convert_to_cities(new_expanded_problem[4])
                            bssf = TSPSolution(route)
                            if bssf.cost < self.lowest_cost:
                                self.lowest_cost = min(bssf.cost, self.lowest_cost)
                                bssf_changes += 1
                            num_solutions += 1
                        else:
                            if new_expanded_problem[5] < self.lowest_cost:
                                heapq.heappush(heap, new_expanded_problem)
                                num_states_made += 1
                            else:
                                pruned += 1
                                num_states_made += 1
            else:
                pruned += 1
                num_states_made += 1
            max_length = self.getMax(len(heap), max_length)

        final_time = time.time()
        results = {}
        results['cost'] = self.lowest_cost
        # results['max'] = max_length
        # results['total'] = num_states_made
        # results['pruned'] = pruned
        results['count'] = num_solutions
        results['soln'] = bssf
        results['time'] = final_time - start_time
        return results


    def get_fancy_reduced_matrix(self, next_city_to_visit, given_tuple):
        self._fancy_count = self._fancy_count + 1
        print('fancy_reduced: ' + str(self._fancy_count))
        # copy to avoid changes to original
        # O(1) time, O(n^2) space
        matrix = given_tuple[3]

        # 0 - val to use for sorting
        # 1 - current node
        # 2 - the rest of the nodes to still visit
        # 3 - reduced matrix for current node
        # 4 - path to current spot
        # 5 - total cost of current path (current node)

        city_copy = deepcopy(given_tuple)
        matrix = matrix.copy()
        sum = 0
        initial_cost = matrix[city_copy[1]._sub_route_index][next_city_to_visit._sub_route_index]

        # BEGIN REDUCING MATRIX
        matrix[city_copy[1]._sub_route_index] = np.inf
        matrix[:, next_city_to_visit._sub_route_index] = np.inf
        matrix[next_city_to_visit._sub_route_index][city_copy[1]._sub_route_index] = np.inf

        for row in range(matrix.shape[0]):
            row_min = np.min(matrix[row])
            if not np.isinf(row_min):
                matrix[row] = matrix[row] - row_min
                sum += row_min

        for col in range(matrix.shape[1]):
            column_min = np.min(matrix[:, col])
            if not np.isinf(column_min):
                matrix[:, col] = matrix[:, col] - column_min
                sum += column_min

        # O(n) time, O(1) space to delete it
        # visits all the cities
        places_to_go = city_copy[2]
        places_to_go = self.delete_city_by_id(places_to_go, next_city_to_visit)

        updated_cost = city_copy[5] + initial_cost + sum

        result = ((self.get_value(city_copy[5], city_copy[4]), next_city_to_visit, places_to_go, matrix,
                   city_copy[4] + [next_city_to_visit._sub_route_index], updated_cost))

        return result

    def get_fancy_init_reduced_matrix(self, city_list, linking_city):
        # linking_city = city_list[len(city_list)-1]
        # city_list = city_list[:len(city_list)-1]
        print('fancy_init')
        reduced_matrix = np.full((len(city_list), len(city_list)), fill_value=np.inf)
        # O(n)
        for origin_index, city in enumerate(city_list):
            for dest_index, dest_city in enumerate(city_list):
                if origin_index == dest_index:
                    continue
                distance = city.costTo(dest_city)
                reduced_matrix[origin_index][dest_index] = distance

        for i in range(len(reduced_matrix)):
            reduced_matrix[i][0] = city_list[i].costTo(linking_city)

        sum = 0
        # O(n^2) to go through every cell in the matrix
        for row in range(reduced_matrix.shape[0]):
            row_min = np.min(reduced_matrix[row])
            reduced_matrix[row] = reduced_matrix[row] - row_min
            sum += row_min

        for col in range(reduced_matrix.shape[1]):
            column_min = np.min(reduced_matrix[:, col])
            reduced_matrix[:, col] = reduced_matrix[:, col] - column_min
            sum += column_min

        return reduced_matrix, sum
