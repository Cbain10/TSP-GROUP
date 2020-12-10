def iterative_TSP_solver(self, cities, time_allowance, time_taken, Lines=None):
    if Lines is None:
        Lines = []

    start_time = time.time()
    # if only one city, return that city
    if len(cities) == 1:
        return cities

    L_side = cities[0:math.floor(len(cities) / 2)]
    r_most_point = L_side[len(L_side) - 1]

    R_side = cities[math.floor(len(cities) / 2):]
    l_most_point = R_side[0]

    # Recursively run iterative_TSP_solver on any arrays larger than 1.
    # Run combine_Routes on arrays whose size is equal to 1 and on return back up recursion stack.
    if (len(L_side) == 1 and len(R_side) == 1):
        return self.combine_routes(L_side, r_most_point, R_side, l_most_point, Lines)
    elif (len(L_side) == 1 and len(R_side) > 1):
        return self.combine_routes(L_side, r_most_point, self.iterative_TSP_solver(R_side, time_allowance,
                                                                                   time_taken + time.time() - start_time),
                                   l_most_point, Lines)
    elif (len(L_side) > 1 and len(R_side) == 1):
        return self.combine_routes(
            self.iterative_TSP_solver(L_side, time_allowance, time_taken + time.time() - start_time, Lines),
            r_most_point, R_side, l_most_point, Lines)
    else:
        return self.combine_routes(
            self.iterative_TSP_solver(L_side, time_allowance, time_taken + time.time() - start_time, Lines),
            r_most_point,
            self.iterative_TSP_solver(R_side, time_allowance, time_taken + time.time() - start_time, Lines),
            l_most_point, Lines)


def combine_routes(self, l_route, r_most_city, r_route, l_most_city, Lines):
    if len(l_route) == 1 and len(r_route) == 1:
        p1 = QPointF(l_route[0]._x, l_route[0]._y)
        p2 = QPointF(r_route[0]._x, r_route[0]._y)
        Lines.append(QLineF(p1, p2))

    return_route = l_route + r_route
    return return_route