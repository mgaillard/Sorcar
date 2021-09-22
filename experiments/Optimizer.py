import math
import random
from timeit import default_timer
from itertools import permutations
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy import linalg, spatial
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from matplotlib import ticker

class OptimizationHistory:
    """ Register evaluations of the cost function as optimization is happening """

    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def get_history(self):
        return np.array(self.history)

    def add_sample(self, xk, state=None):
        """
        Callback for scipy.optimize.minimize()
        Parameter xk is the current parameter vector 
        """
        # Note that we make a deep copy of xk
        self.history.append(np.copy(xk))


# value function + location + sort them by value
class OptimizationAcceptedPointList:
    """
    Register the list of accepted points when running the basinhopping global optimization algorithm
    """

    def __init__(self):
        # List of tuples: (point, f_value)
        self.points = []

    def reset(self):
        self.points = []

    def sort_points(self):
        # Sort according to the function value (increasing)
        self.points.sort(key=lambda x: x[1])

    def is_unique_point(self, threshold=1e-4):
        """
        Compute the maximum Euclidean distance between N pairs of random points
        If the maximum distance is below a threshold, then points are all representing a unique point
        """
        max_dist = 0.0
        for i in range(len(self.points)):
            # Pick a random other point from the distribution
            j = random.randrange(len(self.points))
            d = np.linalg.norm(self.points[i][0] - self.points[j][0])
            max_dist = max(max_dist, d)
        
        return (max_dist <= threshold)

    def remove_duplicate_samples(self, threshold=1e-2):
        """
        Remove sample points that are duplicates of each others
        Two points are considered duplicates if they are within a certain distance 
        Use a KD tree to make this function faster
        Warning: if the points have more than 20 dimensions it can be slower than brute force
        """
        # Create a new array containing only the new points
        new_points = []

        start_time = default_timer()

        # for i in range(len(self.points)):
        #     # Whether point i is unique
        #     is_point_i_unique = True
        #     for j in range(i + 1, len(self.points)):
        #         # Compute the distance between the two points
        #         d = np.linalg.norm(self.points[i][0] - self.points[j][0])
        #         if d < threshold:
        #             is_point_i_unique = False
        #     if is_point_i_unique:
        #         new_points.append(self.points[i])

        kd_tree = spatial.KDTree(self.get_points())
        pairs = kd_tree.query_pairs(threshold)

        # List points that are duplicates and need to be removed
        indices_to_keep = set(range(len(self.points)))
        for (i, j) in pairs:
            if (i in indices_to_keep and j in indices_to_keep):
                # Remove j from the set of points to keep
                indices_to_keep.remove(j)
        
        # Move points that we keep to a new array
        for i in indices_to_keep:
            new_points.append(self.points[i])

        end_time = default_timer()
        print('Removing duplicate samples: {} s'.format(end_time - start_time))

        # Replace points with the new list within duplicates
        self.points = new_points

    def nearest_point(self, x):
        """ Return the nearest point to a query point """
        nearest = self.points[0][0]
        nearest_distance = np.linalg.norm(nearest - x)
        for i in range(len(self.points)):
            dist = np.linalg.norm(self.points[i][0] - x)
            if dist < nearest_distance:
                nearest = self.points[i][0]
                nearest_distance = dist
        return np.copy(nearest)

    def farthest_point(self, x):
        """ Return the farthest point to a query point """
        farthest = self.points[0][0]
        farthest_distance = np.linalg.norm(farthest - x)
        for i in range(len(self.points)):
            dist = np.linalg.norm(self.points[i][0] - x)
            if dist > farthest_distance:
                farthest = self.points[i][0]
                farthest_distance = dist
        return np.copy(farthest)

    def most_delta_change_point(self, x):
        """
        Return the point with changes in parameters that are the closest to a fixed delta change
        For example: we add +1.0 to all parameters, in this case the delta is 1.0
        We look for the point whose range in delta is the smallest among all parameters
        For example in 2D: we look at deltaX and deltaY and compute the range max(deltaX, deltaY) - min(deltaX, deltaY)
                           we select the point whose range is the lowest
        Geometric interpretation: closest to the line starting at x0 with direction vector (1, 1)
        """
        best = self.points[0][0]
        best_range = np.amax(best - x) - np.amin(best - x)
        for i in range(len(self.points)):
            curr_range = np.amax(self.points[i][0] - x) - np.amin(self.points[i][0] - x)
            if curr_range < best_range:
                best = self.points[i][0]
                best_range = curr_range
        return np.copy(best)

    def most_proportional_change_point(self, x):
        """
        Return the point with changes in parameters that are the closest to a proportional change
        For example: we multiply all parameters by 2.0, in this case the multiplier is 2.0
        We look for the point whose range in proportion is the smallest among all parameters
        For example in 2D: we look at pX=X_2-X_1 and pY=Y_2-Y_1 and compute the range max(pX, pY) - min(pX, pY)
                           we select the point whose range is the lowest
        Warning: if the starting configuration has one coordinate that is zero, we ignore this coordinate
        Geometric interpretation: closest to the line starting from the origin and going to x0
        """
        def proportionality_range(x, v):
            array_x = np.array(x)
            # Divide only where the denominator is non null
            divide_max = np.divide(v, array_x, out=np.full_like(v, -np.inf), where=(array_x!=0.0))
            divide_min = np.divide(v, array_x, out=np.full_like(v, np.inf), where=(array_x!=0.0))
            # Return the range of proportionality coefficients
            return np.amax(divide_max) - np.amin(divide_min)

        best = self.points[0][0]
        best_range = proportionality_range(x, best)
        for i in range(len(self.points)):
            curr_range = proportionality_range(x, self.points[i][0])
            if curr_range < best_range:
                best = self.points[i][0]
                best_range = curr_range
        return np.copy(best)

    def least_change_on_axis_point(self, x, axis):
        """
        Return the with the least absolute delta on one axis
        For example: the solution point with the smallest change on X axis
        """
        nearest = self.points[0][0]
        nearest_distance = abs(nearest[axis] - x[axis])
        for i in range(len(self.points)):
            dist = abs(self.points[i][0][axis] - x[axis])
            if dist < nearest_distance:
                nearest = self.points[i][0]
                nearest_distance = dist
        return np.copy(nearest)

    def get_points(self):
        # Extract points only
        points = []
        for p in self.points:
            points.append(p[0])
        return np.array(points)

    def __find_medoids_and_order(self, points, kmedoid_n_clusters):
        """
        Find medoids of a set of points and
        order them in such a way that minimizes the length of the path from the first to the last point
        """

        # If there are more points than the target number of clusters, run KMedoid to reduce the number of points
        if len(points) > kmedoid_n_clusters:
            # For all clusters, run KMedoid and replace the points by the medoids
            kmedoids = KMedoids(n_clusters=kmedoid_n_clusters,
                                metric='euclidean',
                                method='alternate',
                                init='k-medoids++',
                                max_iter=300,
                                random_state=None).fit(points)
            medoids = kmedoids.cluster_centers_
        else:
            # No need to reduce the number of points
            medoids = points

        # Order points to minimize distance between them
        points_ordering = shortest_hamiltonian_path_bruteforce(medoids)
        # Return the points with the best ordering
        return medoids[points_ordering]

    def cluster_and_order_points(self, visualize=False):
        """
        Cluster points with DBSCAN and order them
        Points must not be empty
        Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        """
        start_time = default_timer()

        points = self.get_points()
        db = DBSCAN(eps=0.5, min_samples=1).fit(points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        unique_labels = set(labels)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        cluster_points = []
        noise_points = []

        # Get points per cluster
        for k in unique_labels:
            class_member_mask = (labels == k)
            if k == -1:
                # Noise points
                noise_points = points[class_member_mask]
            elif k >= 0:
                # Actual cluster point
                points_in_cluster = points[class_member_mask]
                cluster_representatives_ordered = self.__find_medoids_and_order(points_in_cluster, 8)
                points_in_cluster_ordered = order_points_on_line_segment(points_in_cluster, cluster_representatives_ordered)
                cluster_points.append(points_in_cluster_ordered)
                print('Cluster {} # of points: {}'.format(k, len(points_in_cluster_ordered)))        

        # Noise is the last cluster
        if len(noise_points) > 0:
            cluster_points.append(noise_points)

        end_time = default_timer()
        print('Clustering and ordering of points: {} s'.format(end_time - start_time))

        if visualize:
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(cluster_points))]
            for i in range(len(cluster_points)):
                plt.plot(cluster_points[i][:, 0], cluster_points[i][:, 1], marker='o', color=tuple(colors[i]), linestyle='-')
        
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()

        return cluster_points

    def basinhopping_callback(self, x, f, accept):
        """
        Callback for scipy.optimize.basinhopping()
        Parameter x is the current parameter vector
        Parameter f is the value of the function
        Parameter accept is True if it's an accepted minimum
        """
        if accept:
            # Note that we make a deep copy of x
            self.points.append((np.copy(x), f))


class Optimizer:
    """
    An optimizer that returns different configurations
    """

    class _BasinhoppingTakeStepBounds:
        """
        Specific class for overriding take_step in the basinhopping method
        Normalize the coordinates between the bounds
        """
        def __init__(self, bounds_range, stepsize=0.5):
            self.bounds_range = bounds_range
            self.stepsize = stepsize
            self.rng = np.random.default_rng()

        def __call__(self, x):
            # The stepsize can be changed by the basinhopping method
            s = self.stepsize
            # Generate step in [-1.0; 1.0] when s=1.0
            step = self.rng.uniform(-s, s, x.shape)
            if self.bounds_range is not None:
                # Remap the value with the range
                x = x + step * self.bounds_range
            else:
                # No remapping needed if there are no bounds
                x = x + step
            return x

    def __init__(self, function, bounds, x0):
        """
        Initialize the optimizer
        """
        # Function to optimize
        self.function = function
        # Bounds of parameters in SciPy format and in Numpy format
        self.bounds = bounds
        if self.bounds is not None:
            self.bounds_xmin = np.zeros_like(self.bounds.lb)
            self.bounds_xmax = np.zeros_like(self.bounds.ub)
            self.bounds_range = np.zeros_like(self.bounds.ub)
            for i in range(len(self.bounds.lb)):
                self.bounds_xmin[i] = self.bounds.lb[i]
                self.bounds_xmax[i] = self.bounds.ub[i]
                if (self.bounds.ub[i] < np.inf) and (self.bounds.lb[i] > -np.inf):
                    self.bounds_range[i] = self.bounds.ub[i] - self.bounds.lb[i]
                else:
                    self.bounds_range[i] = 1.0
        else:
            self.bounds_xmin = None
            self.bounds_xmax = None
            self.bounds_range = None
        # Function to take normalized steps with the basinhopping method
        self.__basinhopping_take_step_bounds = self._BasinhoppingTakeStepBounds(self.bounds_range)
        # Initial point for optimization
        self.x0 = x0
        # Budget of function evaluation left
        self.budget = 0
        # Total time (in seconds) spent optimizing
        self.total_time = 0.0
        # Best optimal point found so far (by default: the initial point)
        self.best_optimal = x0
        self.best_value = self.function.evaluate(x0)
        # Hessian of the best 
        self.best_hessian = None
        # Set of optimal points found when exploring the optimal region
        self.optimal_points = None

    def get_total_time(self):
        """ Return total optimization time """
        return self.total_time

    def get_budget(self):
        """ Return the remaining budget for function evaluation """
        return self.budget

    def increase_budget(self, additional_budget):
        """
        Increase the budget of function evaluation
        """
        self.budget = self.budget + additional_budget

    def __get_best_minimizer_args(self):
        """
        Return a dictionary with the best arguments for local optimization
        Uses optimizers from SciPy
        """

        if self.bounds is None:
            # No bounds => BFGS
            return {
                'method':'BFGS',
                'jac': self.function.derivative,
                'options': {'gtol': 1e-6, 'disp': False}
            }
        elif self.bounds is not None:
            # Bounds => L-BFGS-B
            return {
                'method':'L-BFGS-B',
                'jac': self.function.derivative,
                'bounds': self.bounds,
                'options': {'gtol': 1e-6, 'disp': False}
            }

    def __update_time_and_budget(self, elapsed_time, res):
        """
        Update the total time spent optimizing
        Update the budget of function evaluation
        """
        self.total_time = self.total_time + elapsed_time
        self.budget = max(self.budget - res.nfev, 0)

    def __update_best_optimal(self, res):
        """
        Private function to update the best optimal point from an OptimizeResult object
        """
        # Only if the optimization was successful, or if the successful flag is not set
        if (hasattr(res, 'success') and res.success) or not hasattr(res, 'success'):
            # Update the best optimal point
            if res.fun < self.best_value:
                print('Found a better optimal point: {}'.format(res.x))
                self.best_optimal = res.x
                self.best_value = res.fun
                if self.function.is_hessian_active():
                    # Compute the Hessian
                    self.best_hessian = self.function.hessian(res.x)
                elif hasattr(res, 'hess') and res.hess is not None:
                    # Use the approximated Hessian if Hessian is not available
                    self.best_hessian = res.hess
                elif hasattr(res, 'hess_inv') and res.hess_inv is not None:
                    # Use the pseudo inverse of the inverse of the Hessian (when using BFGS)
                    # For L-BFGS-B, which estimates the Hessian inverse implicitly
                    if hasattr(res.hess_inv, 'todense') and callable(res.hess_inv.todense):
                        hess_inv = res.hess_inv.todense()
                    # For BFGS, which estimates the Hessian inverse explicitly
                    else:
                        hess_inv = res.hess_inv
                    self.best_hessian = linalg.pinvh(hess_inv)
                else:
                    self.best_hessian = None

    def __is_best_optimal_unique(self, threshold=1e-2):
        """
        Check if there is potentially a set of 
        Look at the eigen values of the Hessian around the current best optimal point,
        if at least one eigen value is close to zero,
        then the solution may not be unique and the function is underdetermined
        Hessian matrices are symmetric, so their eigenvalues are real numbers
        If all eigenvalues are positive, the Hessian is positive-definite => local minimum
        If all eigenvalues are negative, the Hessian is negative-definite => local maximum
        If eigenvalues are mixed (positive, negative) => saddle point
        If either eigenvalue is zero, the Hessian needs more investigation
        """
        if self.best_hessian is not None:
            eigenvalues = linalg.eigvalsh(np.array(self.best_hessian))
            for eigenvalue in eigenvalues:
                if abs(eigenvalue) < threshold:
                    # If at least one eigenvalue is close to zero, the optimal point may not be unique
                    return False
            return True
        else:
            # If the Hessian at the best optimal point is not available,
            # we cannot decide and thus return False
            # so that we better investigate with the local search
            return False

    def __basinhopping_accept_test_bounds(self, **kwargs):
        """
        Specific function for overriding accept_test in the basinhopping method
        Only accept steps that are in within the bounds
        """
        if self.bounds is not None:
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.bounds_xmax))
            tmin = bool(np.all(x >= self.bounds_xmin))
            return tmax and tmin
        else:
            return True

    def local_optimization(self):
        """
        Run local optimization on the problem
        Uses L-BFGS-B if gradient and bounds are available
        """
        optim_history = OptimizationHistory()
        optim_history.add_sample(self.x0)

        start_time = default_timer()
        minimizer_kwargs = self.__get_best_minimizer_args()
        res = minimize(self.function.evaluate,
                       self.x0,
                       callback=optim_history.add_sample,
                       **minimizer_kwargs)
        end_time = default_timer()
        print('Local optimization time: {} s'.format(end_time - start_time))
        # Updates after optimization
        self.__update_best_optimal(res)
        self.__update_time_and_budget(end_time - start_time, res)

    def global_optimization(self):
        """
        Run global optimization on the problem
        Use basinhopping method
        Starts from the already known best optimal point, if it has been found by local search
        """
        optim_points = OptimizationAcceptedPointList()
        
        start_time = default_timer()
        minimizer_kwargs = self.__get_best_minimizer_args()
        res = basinhopping(self.function.evaluate,
                           self.best_optimal,
                           niter=self.budget,
                           T=1.0,         # Temperature for global optimization is the default value: 1.0
                           stepsize=0.25, # Step size for global optimization is one quarter of the range
                           minimizer_kwargs=minimizer_kwargs,
                           take_step=self.__basinhopping_take_step_bounds,
                           accept_test=self.__basinhopping_accept_test_bounds,
                           callback=optim_points.basinhopping_callback,
                           interval=50,
                           disp=None)
        end_time = default_timer()
        print('Global optimization time: {} s'.format(end_time - start_time))
        # Updates after optimization
        self.__update_best_optimal(res)
        self.__update_time_and_budget(end_time - start_time, res)
        # Check that the optimal point is unique
        if not optim_points.is_unique_point():
            print('Warning: Global optimization yielded multiple optima! The solution may be undetermined.')
            # Retain the optimal points
            optim_points.sort_points()
            optim_points.remove_duplicate_samples()
            self.optimal_points = optim_points

    def explore_optimality_region(self):
        """
        Explore the region of optimality of the function to optimize
        Call this function when the problem is underdetermined to find other optimal points
        The temperature is set to a lower value 
        """
        optim_points = OptimizationAcceptedPointList()

        start_time = default_timer()
        minimizer_kwargs = self.__get_best_minimizer_args()
        res = basinhopping(self.function.evaluate,
                           self.best_optimal,
                           niter=self.budget,
                           T=0.1,         # Temperature for exploration is less than the default value: 0.1
                           stepsize=0.05, # Step size for exploration is 5% of the range
                           minimizer_kwargs=minimizer_kwargs,
                           take_step=self.__basinhopping_take_step_bounds,
                           accept_test=self.__basinhopping_accept_test_bounds,
                           callback=optim_points.basinhopping_callback,
                           interval=50,
                           disp=None)
        end_time = default_timer()
        print('Optimal region exploration time: {} s'.format(end_time - start_time))

        # Update the budget
        self.__update_time_and_budget(end_time - start_time, res)

        # Retain the optimal points
        optim_points.sort_points()
        optim_points.remove_duplicate_samples()
        self.optimal_points = optim_points

    def optimize(self, budget):
        """
        Optimize the function in a two stage method:
        1) Local optimization
        2) If the solution is determined, use the rest of the budget to find the global optimum
        3) If the solution is underdetermined, use the rest of the budget to explore the region of optimality
        Parameter budget gives the number of maximum function evaluation before giving an answer
        """
        self.budget = budget
        print('Launching local optimization...')
        self.local_optimization()
        # If there is budget left, we can spend it on global optimization or exploration
        if self.budget > 0:
            sol_determined = self.__is_best_optimal_unique()
            if sol_determined:
                # Global optimization
                print('Launching global optimization...')
                self.global_optimization()
            else:
                # Explore the region of optimality
                print('Exploring the local region...')
                self.explore_optimality_region()
        return self.best_optimal

    def plot_2D(self):
        """
        Plot the function in 2D
        """
        if self.function.dimensionality==2:
            # All points
            all_points = np.array([self.x0])
            # Interesting points to display
            interesting_points = [
                (self.x0, 'r*'),          # Starting point with a red star
                (self.best_optimal, 'g*') # Global optimum with a green star
            ]

            if self.optimal_points is not None:
                o = self.optimal_points
                all_points = o.get_points()
                # Nearest optimum with a magenta +
                interesting_points.append((o.nearest_point(self.x0), 'mP'))
                # Farthest optimum with a magenta hexagon
                interesting_points.append((o.farthest_point(self.x0), 'mH'))
                # Most delta change optimum with a yellow square
                interesting_points.append((o.most_delta_change_point(self.x0), 'ys'))
                # Most proportional change optimum with a yellow octagon
                interesting_points.append((o.most_proportional_change_point(self.x0), 'y8'))
                # Least change in X coordinate optimum with a white X
                interesting_points.append((o.least_change_on_axis_point(self.x0, 0), 'wX'))
                # Least change in Y coordinate optimum with a white v
                interesting_points.append((o.least_change_on_axis_point(self.x0, 1), 'wv'))

            plot_function_contour_with_samples(self.function,
                                               self.bounds,
                                               all_points,
                                               show_arrows=False,
                                               interesting_points=interesting_points)


def plot_function_contour_with_samples(func, bounds, path, show_arrows, interesting_points=[]):
    """
    2D contour plot of the function with optimization path
    Source: http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
    """
    transpose_path = path.T

    resolutionX = 50
    resolutionY = 50

    if bounds is not None:
        xmin = bounds.lb[0]
        xmax = bounds.ub[0]
        ymin = bounds.lb[1]
        ymax = bounds.ub[1]
    else:
        xmin = -5.0
        xmax = 5.0
        ymin = -5.0
        ymax = 5.0

    x = np.linspace(xmin, xmax, resolutionX)
    y = np.linspace(ymin, ymax, resolutionY)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((resolutionX, resolutionY))
    for i in range(resolutionX):
        for j in range(resolutionY):
            Z[i, j] = func.evaluate([X[i, j], Y[i, j]])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Contour plot of the function
    contour_lines = 20 # Number of contour lines
    ax.contourf(X, Y, Z, contour_lines, cmap=plt.cm.jet)
    contours = ax.contour(X, Y, Z, contour_lines, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    # Path with arrows
    if show_arrows:
        ax.quiver(transpose_path[0,:-1],
                  transpose_path[1,:-1],
                  transpose_path[0,1:] - transpose_path[0,:-1],
                  transpose_path[1,1:] - transpose_path[1,:-1],
                  scale_units='xy',
                  angles='xy',
                  scale=1,
                  color='k')
    else:
        ax.scatter(transpose_path[0,:-1],
                   transpose_path[1,:-1],
                   c='black')

    # Starting point in red
    for interesting_point in interesting_points:
        p = interesting_point[0] # 2D point
        c = interesting_point[1] # Color and marker type
        ax.plot(p[0], p[1], c, markersize=10)

    # Axes labels
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    plt.show()


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


def shortest_hamiltonian_path_bruteforce(points):
    """
    Find the order of points that minimizes the total Euclidean distance between them
    Warning: brute force
    """

    num_points = len(points)
    distance_matrix = spatial.distance.cdist(points, points, metric='euclidean')

    # Store all indexes
    ordered_vertices = []
    for i in range(num_points):
        ordered_vertices.append(i)
 
    # Store minimum weight Hamiltonian path
    min_path = hamiltonian_path_objective_function(distance_matrix, np.array(ordered_vertices))
    best_points = ordered_vertices

    next_permutation = permutations(ordered_vertices)
    for permutation in next_permutation:
        # Measure current permutation
        current_path_length = hamiltonian_path_objective_function(distance_matrix, np.array(permutation))
        # Update minimum
        if current_path_length < min_path:
            min_path = current_path_length
            best_points = permutation
         
    return np.asarray(best_points)


def point_line_projection(p, a, b):
    """
    Projection of point P on the line (AB)
    Return the parametric coordinate U
    """
    vec_ap = p - a
    vec_ab = b - a
    norm_sq = np.dot(vec_ab, vec_ab)

    if norm_sq <= 0.0:
        return 0.0

    return np.dot(vec_ap, vec_ab) / norm_sq


def point_multi_line_segment_projection(p, line_segment_points):
    """
    Projection of a point P on a multi line segment [Q0, Q1, Q2, ..., Qn-1]
    Return the parametric coordinate U between 0.0 and N-1
    """

    # Squared distance to the projection of point P on the line segment
    minimum_distance_sq = np.inf
    # Parametric coordinate of point P on the line segment
    minimum_u = -1.0

    for i in range(len(line_segment_points) - 1):
        # Start point of the line segment
        a = line_segment_points[i]
        # End point of the line segment
        b = line_segment_points[i + 1]
        # Projection on the current line segment
        u = point_line_projection(p, a, b)

        # If the point is before the segment clamp to the segment
        # unless it's the first segment, in this case, we let it go before on the line
        if (u < 0.0) and (i > 0):
            u = 0.0
        # If the point is after the segment clamp to the segment
        # unless it's the last segment, in this case, we let it go after on the line
        elif (u > 1.0) and (i < (len(line_segment_points) - 2)):
            u = 1.0

        # Compute the distance to the current line segment
        c = a * (1.0 - u) + b * u
        d_sq = np.dot(p - c, p - c)

        if d_sq < minimum_distance_sq:
            minimum_distance_sq = d_sq
            # The parametric coordinate is the sum of
            #  - the index of the segment: 0, 1, 2, 3, ...
            #  - the parametric coordinate on that segment in [0, 1]
            minimum_u = i + u

    return minimum_u


def order_points_on_line_segment(points, line_segment_points):
    """
    Sort points to be ordered along a line segment defined by a list of points in an ND space
    """

    points_u_coord = []
    # For each point, we find the parametric coordinates of its projection on the line segment
    for i in range(len(points)):
        u = point_multi_line_segment_projection(points[i], line_segment_points)
        points_u_coord.append((u, i))

    # We sort points according to the parametric coordinates
    points_u_coord.sort(key=lambda x: x[0])

    # Return the points only
    points_order = []
    for pu in points_u_coord:
        points_order.append(pu[1])

    return points[points_order]
