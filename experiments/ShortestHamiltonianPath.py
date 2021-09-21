import numpy as np
from scipy import spatial
from itertools import permutations
from sko.ACA import ACA_TSP
from sko.SA import SA_TSP

def hamiltonian_path_objective_function(distance_matrix, routine):
    """
    The objective function for shortest Hamiltonian path.
    Starts from the first point and goes to the last point, but does not cycle.
    Input: routine
    Return: total distance
    hamiltonian_path_objective_function(distance_matrix, np.arange(num_points))
    """
    num_points, = routine.shape
    cost = sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points - 1)])
    return cost

def shortest_hamiltonian_path(points):
    """
    Find the order of points that minimizes the total Euclidean distance between them
    Only works for a low number of points
    """

    # Order the points with TSP
    num_points = len(points)
    distance_matrix = spatial.distance.cdist(points, points, metric='euclidean')
    objective_func = lambda routine : hamiltonian_path_objective_function(distance_matrix, routine)
    
    # tsp = ACA_TSP(func=objective_func, n_dim=num_points, size_pop=50, max_iter=200, distance_matrix=distance_matrix)
    tsp = SA_TSP(func=objective_func, x0=range(num_points), T_max=100, T_min=1, L=10 * num_points)
    best_points, best_distance = tsp.run()
    
    return best_points
