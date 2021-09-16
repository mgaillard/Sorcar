import math
import random
from timeit import default_timer
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy import linalg
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

        # Sort points per increasing f value
        optim_points.sort_points()
        # Retain the optimal points
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
        sol_determined = self.__is_best_optimal_unique()
        if sol_determined:
            # Global optimization
            print('Launching global optimization...')
            self.global_optimization()
        else:
            # Explore the region of optimality
            print('Exploring the local region...')
            self.explore_optimality_region()

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
