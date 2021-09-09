import math
from timeit import default_timer
import numpy as np
from casadi import *
from scipy.optimize import minimize, basinhopping
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import ticker

# TODO: Improve the accept bounds in the global optimization
# TODO: make the whole algorithm work with bounded functions
#        - Use L-BFGS-B for local optimization
#        - Use accept_test argument in basinhopping to set bounds
#          See last example of: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
# TODO: import more Blender functions with two variables (two cube stack)
# TODO: make it possible to not provide the Hessian in Casadi Function
# TODO: benchmark the time needed for computing the gradient vs the Hessian of a function
# TODO: run K-Medoid on the set of all optimal points
# TODO: find an order for the medoids
# TODO: try nonlinear-PCA on the set of points
# TODO: show a slider (1D or 2D) for exploring the solution set, and by changing it, show the position on the 2D plot
# TODO: try trust region methods for local optimization
# TODO: try to see if an increase in the stepsize improves the spread of samples
# TODO: in priority, try to change the basinhopping instead of reinventing the wheel, bias steps with the Hessian
# TODO: animation of the optimization plot
# TODO: minimize the list of functions with different optimizers and show different animations

class OptimizationHistory:
    """ Register evaluations of the cost function as optimization is happening """

    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def get_history(self):
        return np.array(self.history)

    def add_sample(self, xk):
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
            for j in range(len(self.points)):
                if i < j:
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
        Warning: if the starting configuration has one coordinate that is zero, this may fail
        Geometric interpretation: closest to the line starting from the origin and going to x0
        """
        best = self.points[0][0]
        best_range = np.amax(best / x) - np.amin(best / x)
        for i in range(len(self.points)):
            curr_range = np.amax(self.points[i][0] / x) - np.amin(self.points[i][0] / x)
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


class CasadiFunction:
    """
    Casadi wrapper for a function to optimize.
    Jacobian and Hessian matrices are computed using automatic differentiation
    """

    def __init__(self):
        """
        Initialize the function to optimize and its gradient
        """
        self.dimensionality = 0
        self.func = None
        self.grad_func = None
        self.hess_func = None

    def set_expression(self, expression, symbols):
        """
        Set the expression of the function to optimize and compute its gradient/Hessian
        """
        function_options = {
            'jit': False, # By default JIT is not activated
        }
        # Number of dimensions of the function to optimize
        self.dimensionality = symbols.shape[0]
        # Build CasADi Function for evaluating the function to optimize
        self.func = Function('f', [symbols], [expression], function_options)
        # Build CasADi Function for evaluating the gradient of the function to optimize
        grad_expression = gradient(expression, symbols)
        self.grad_func = Function('g', [symbols], [grad_expression], function_options)
        # Buils CasADi Function for evaluating the Hessian of the function to optimize
        hess_expression, g = hessian(expression, symbols)
        self.hess_func = Function('h', [symbols], [hess_expression], function_options)

    def evaluate(self, x):
        # Check that the input has the right dimensionality
        if len(x) == self.dimensionality:
            # Call the function with the parameters
            result = self.func.call([x])
            # Convert the output to float
            return float(result[0])
        # By default, the output is None
        return None

    def derivative(self, x):
        # Check that the input has the right dimensionality
        if len(x) == self.dimensionality:
            # Call the function with the parameters
            results = self.grad_func.call([x])
            # Convert the output to float
            output = []
            for i in range(self.dimensionality):
                output.append(float(results[0][i]))
            return output
        # By default, the output is None
        return None

    def hessian(self, x):
        # Check that the input has the right dimensionality
        if len(x) == self.dimensionality:
            # Call the function with the parameters
            results = self.hess_func.call([x])
            # Convert the output to float
            output = []
            for i in range(self.dimensionality):
                row = []
                for j in range(self.dimensionality):
                    row.append(float(results[0][i, j]))
                output.append(row)
            return output
        # By default, the output is None
        return None

    def taylor(self, x, p):
        """
        Second order approximation of the function using the gradient and the Hessian
        """
        dx = np.array(p) - np.array(x)
        f = np.array(self.evaluate(x))
        g = np.array(self.derivative(x))
        H = np.array(self.hessian(x))

        approximation = f + np.dot(g, dx) + 0.5 * np.dot(dx, np.matmul(H, dx))

        return approximation

    def generate_code(self, filename):
        """
        Generate C code associated to the functions for later JIT compilation
        """
        generator = CodeGenerator(filename)
        generator.add(self.func)
        generator.add(self.grad_func)
        generator.add(self.hess_func)
        generator.generate()

    def load_and_jit_compile_code(self, filename):
        """
        Load a previously saved function
        JIT compile it using the embedded clang in Casadi
        """
        importer = Importer(filename, 'clang')
        self.func = external('f', importer)
        self.grad_func = external('g', importer)
        self.hess_func = external('h', importer)
        # Infer the dimensionality from the function input size
        self.dimensionality = self.func.size1_in(0)

    def load_binary_code(self, filename):
        """
        Load a previously saved function already compiled
        """
        self.func = external('f', filename)
        self.grad_func = external('g', filename)
        self.hess_func = external('h', filename)
        # Infer the dimensionality from the function input size
        self.dimensionality = self.func.size1_in(0)


class Optimizer:
    """
    An optimizer that returns different configurations
    """

    def __init__(self, function, bounds, x0):
        """
        Initialize the optimizer
        """
        # Function to optimize
        self.function = function
        # Bounds of parameters
        self.bounds = bounds
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
                self.best_hessian = self.function.hessian(res.x)
                # TODO: Use the approximated Hessian if it's not available, otherwise compute it directly
                # self.best_hessian = res.hess

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

    def local_optimization(self):
        """
        Run local optimization on the problem
        Uses L-BFGS-B if gradient and bounds are available
        """
        optim_history = OptimizationHistory()
        optim_history.add_sample(self.x0)

        start_time = default_timer()
        res = minimize(self.function.evaluate,
                       self.x0,
                       method='L-BFGS-B',
                       jac=self.function.derivative,
                       # hess=self.function.hessian,
                       bounds=self.bounds,
                       callback=optim_history.add_sample,
                       options={'gtol': 1e-6, 'disp': True})
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
        start_time = default_timer()
        minimizer_kwargs = {
            'method':'BFGS',
            'jac': self.function.derivative,
            'bounds': self.bounds
        }
        res = basinhopping(self.function.evaluate,
                           self.best_optimal,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=self.budget,
                           disp=None)
        end_time = default_timer()
        print('Global optimization time: {} s'.format(end_time - start_time))
        # Updates after optimization
        self.__update_best_optimal(res)
        self.__update_time_and_budget(end_time - start_time, res)

    def explore_optimality_region(self):
        """
        Explore the region of optimality of the function to optimize
        Call this function when the problem is underdetermined to find other optimal points
        """
        optim_points = OptimizationAcceptedPointList()

        start_time = default_timer()
        minimizer_kwargs = {
            'method':'BFGS',
            'jac': self.function.derivative,
            'bounds': self.bounds
        }
        res = basinhopping(self.function.evaluate,
                           self.best_optimal,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=self.budget,
                           callback=optim_points.basinhopping_callback,
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


def generate_rosen():
    """
    2D Rosenbrock function
    a = 1.0
    b = 100.0
    Optimimum: [1.0, 1.0]
    """
    x = SX.sym('x', 2)
    expr = pow(1.0 - x[0], 2) + 100.0*pow(x[1] - x[0]*x[0], 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-2.0, 2.0), (-1.0, 3.0)],
        'starting_point': [1.3, 0.7]
    }


def generate_underdetermined_linear():
    """
    Underdetermined function: z = x + y
    f(x,y) = (z - 1.0)*(z - 1.0)
    """
    x = SX.sym('x', 2)
    # Function
    z = x[0] + x[1]
    # We want to fit z so that z = 1.0
    expr = pow(z - 1.0, 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-2.0, 2.0), (-2.0, 2.0)],
        'starting_point': [1.3, 0.7]
    }


def generate_underdetermined_circle():
    """
    Underdetermined function: squared distance to circle of radius 1.0
    """
    x = SX.sym('x', 2)
    # Signed distance to circle of radius 1.0
    d = norm_2(x) - 1.0
    # Squared distance
    expr = pow(d, 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-2.0, 2.0), (-2.0, 2.0)],
        'starting_point': [1.3, 0.7]
    }


def generate_underdetermined_disk():
    """
    Underdetermined function: zero ellipsoid disk with square distance to it
    """
    x = SX.sym('x', 2)
    # Signed distance to ellipsoid circle of radius 1.0
    d = norm_2(x / SX([2.0, 1.0])) - 1.0
    # Remove the negative portion of the disk
    d_plus = fmax(d, 0.0)
    # Squared distance
    expr = pow(d_plus, 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-3.0, 3.0), (-3.0, 3.0)],
        'starting_point': [1.5, -1.8]
    }


def generate_underdetermined_arm():
    """
    Underdetermined function: 2D robotic arm sliding
    """
    x = SX.sym('x', 2)
    # One arm is 1.0 long, the second arm is 2.0 long
    y = 1.0 * sin(x[0]) + 2.0 * sin(x[1])
    # Try to keep the arm on the X axis (i.e. y=0)
    expr = pow(y, 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-3.14, 3.14), (-3.14, 3.14)],
        'starting_point': [0.785, 0.0]
    }


def generate_underdetermined_non_optimal():
    """
    Underdetermined function whose optimum is greater than 0.0
    """
    x = SX.sym('x', 2)
    expr = 1.0 + pow(x[1] - x[0]*x[0], 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-2.0, 2.0), (-10.0, 10.0)],
        'starting_point': [1.3, 0.7]
    }


def generate_sorcar_cube_size_underdetermined():
    """
    Underdetermined function directly exported from Sorcar add-on
    The size of a cube is determined by two variables that are added to each other
    """
    # Build the function
    function = CasadiFunction()
    function.load_and_jit_compile_code('functions/sorcar_cube_size.c')
    return {
        'function': function,
        'bounds': [(0.0, 5.0), (0.0, 5.0)],
        'starting_point': [2.0, 0.0]
    }


def generate_himmelblau():
    """
    Himmelblau function
    (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Has four local minima (up to 1 decimal place):
     * 3.0, 2.0
     * -2.8, 3.1
     * -3.8, -3.3
     * 3.6, -1.8
    """
    x = SX.sym('x', 2)
    expr = pow(x[0]*x[0] + x[1] - 11.0, 2) + pow(x[0] + x[1]*x[1] - 7.0, 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-5.0, 5.0), (-5.0, 5.0)],
        'starting_point': [1.3, 0.7]
    }


def generate_rastrigin():
    """
    Rastigrin function
    20.0 + (x^2 - 10.0*cos(2*pi*x)) + (y^2 - 10.0*cos(2*pi*y))
    """
    x = SX.sym('x', 2)
    expr = 20.0 + pow(x[0]*x[0] - 10.0*cos(2.0*math.pi*x[0]), 2) + pow(x[1]*x[1] - 10.0*cos(2.0*math.pi*x[1]), 2)
    # Build the function
    function = CasadiFunction()
    function.set_expression(expr, x)
    return {
        'function': function,
        'bounds': [(-5.0, 5.0), (-5.0, 5.0)],
        'starting_point': [1.3, 0.7]
    }


def generate_functions():
    """ Generate a list of predefined functions to optimize """
    functions = {
        'rosen': generate_rosen(),
        'underdetermined_linear': generate_underdetermined_linear(),
        'underdetermined_circle': generate_underdetermined_circle(),
        'underdetermined_disk': generate_underdetermined_disk(),
        'underdetermined_arm': generate_underdetermined_arm(),
        'underdetermined_non_optimal': generate_underdetermined_non_optimal(),
        'sorcar_cube_size_underdetermined': generate_sorcar_cube_size_underdetermined(),
        'himmelblau': generate_himmelblau(),
        'rastrigin': generate_rastrigin()
    }
    return functions


def plot_surface_and_taylor(function, point):
    """
    Plot the surface of a function and it's local Taylor approximation around the point x
    """
    func = function['function']
    bounds = function['bounds']

    resolutionX = 50
    resolutionY = 50

    x = np.linspace(bounds[0][0], bounds[0][1], resolutionX)
    y = np.linspace(bounds[1][0], bounds[1][1], resolutionY)
    X, Y = np.meshgrid(x, y)

    # True function
    Z1 = np.zeros((resolutionX, resolutionY))
    # Taylor approximation
    Z2 = np.zeros((resolutionX, resolutionY))
    # Absolute difference
    D = np.zeros((resolutionX, resolutionY))
    for i in range(resolutionX):
        for j in range(resolutionY):
            Z1[i, j] = func.evaluate([X[i, j], Y[i, j]])
            Z2[i, j] = func.taylor(point, [X[i, j], Y[i, j]])
            D[i, j] = abs(Z1[i, j] - Z2[i, j])

    contour_lines = 20 # Number of contour lines
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.contourf(X, Y, Z1, contour_lines, cmap=plt.cm.jet)
    ax1.plot(point[0], point[1], 'r*', markersize=10)
    ax2.contourf(X, Y, Z2, contour_lines, cmap=plt.cm.jet)
    ax2.plot(point[0], point[1], 'r*', markersize=10)
    ax3.contourf(X, Y, D, contour_lines, cmap=plt.cm.jet)
    ax3.plot(point[0], point[1], 'r*', markersize=10)

    plt.show()


def plot_function_contour_with_samples(func, bounds, path, show_arrows, interesting_points=[]):
    """
    2D contour plot of the function with optimization path
    Source: http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
    """
    transpose_path = path.T

    resolutionX = 50
    resolutionY = 50

    x = np.linspace(bounds[0][0], bounds[0][1], resolutionX)
    y = np.linspace(bounds[1][0], bounds[1][1], resolutionY)
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


def main():
    functions = generate_functions()

    functions_to_optimize = [
        'rosen',
        'underdetermined_circle',
        'underdetermined_disk',
        'underdetermined_arm'
    ]

    # Optimization of the functions
    for f in functions_to_optimize:
        optimizer = Optimizer(functions[f]['function'],
                              functions[f]['bounds'],
                              functions[f]['starting_point'])
        optimizer.optimize(400)
        print('Total optimization time: {} s'.format(optimizer.get_total_time()))
        optimizer.plot_2D()


if __name__ == "__main__":
    main()
