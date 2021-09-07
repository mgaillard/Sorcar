import math
from timeit import default_timer
import numpy as np
from casadi import *
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
from matplotlib import ticker

# TODO: animation of the optimization plot
# TODO: minimize the list of functions with different optimizers and show different animations
# TODO: add underdetermined functions with a solution set that is 1D and curved but without a loop
# TODO: try trust region methods for local optimization
# TODO: find interesting samples
#        - the one that keeps proportions in the input parameters
#        - the one with the same delta added to the parameters
#        - the one with the least change in one dimension
# TODO: run K-Medoid on the set of points
# TODO: find an order for the medoids
# TODO: try nonlinear-PCA on the set of points
# TODO: show a slider (1D or 2D) for exploring the solution set, and by changing it, show the position on the 2D plot
# TODO: early stop if the solution is unique (no basin), then switch to global optimization until the budget is over
# TODO: try to see if an increase in the stepsize improve the spread of samples
# TODO: try a function whose optimum is not 0 but a higher value
# TODO: in priority, try to change the basinhopping instead of reinventing the wheel, bias steps with the Hessian
# TODO: make the whole algorithm work with bounded functions
#        - Use L-BFGS-B for local optimization
#        - Use accept_test argument in basinhopping to set bounds
#          See last example of: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

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
        """
        best = self.points[0][0]
        best_range = np.amax(best / x) - np.amin(best / x)
        for i in range(len(self.points)):
            curr_range = np.amax(self.points[i][0] / x) - np.amin(self.points[i][0] / x)
            if curr_range < best_range:
                best = self.points[i][0]
                best_range = curr_range
        return np.copy(best)

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

    def __init__(self, expression, symbols):
        """
        Initialize the function to optimize and its gradient
        """
        # Number of dimensions of the function to optimize
        self.dimensionality = symbols.shape[0]
        # Build CasADi Function for evaluating the function to optimize
        self.func = Function('f', [symbols], [expression])
        # Build CasADi Function for evaluating the gradient of the function to optimize
        grad_expression = gradient(expression, symbols);
        self.grad_func = Function('g', [symbols], [grad_expression])
        # Buils CasADi Function for evaluating the Hessian of the function to optimize
        hess_expression, g = hessian(expression, symbols)
        self.hess_func = Function('h', [symbols], [hess_expression])

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


def generate_rosen():
    """
    2D Rosenbrock function
    a = 1.0
    b = 100.0
    Optimimum: [1.0, 1.0]
    """
    x = SX.sym('x', 2)
    expr = pow(1.0 - x[0], 2) + 100.0*pow(x[1] - x[0]*x[0], 2)
    return {
        'function': CasadiFunction(expr, x),
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
    return {
        'function': CasadiFunction(expr, x),
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
    return {
        'function': CasadiFunction(expr, x),
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
    return {
        'function': CasadiFunction(expr, x),
        'bounds': [(-3.0, 3.0), (-3.0, 3.0)],
        'starting_point': [1.5, -1.8]
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
    return {
        'function': CasadiFunction(expr, x),
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
    return {
        'function': CasadiFunction(expr, x),
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


def plot_function_contour_with_samples(function, path, show_arrows, interesting_points=[]):
    """
    2D contour plot of the function with optimization path
    Source: http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
    """
    func = function['function']
    bounds = function['bounds']
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


def optimization(function):
    """
    Local optimization of a function
    """
    func = function['function']
    bounds = function['bounds']
    x0 = function['starting_point']

    optim_history = OptimizationHistory()
    optim_history.add_sample(x0)

    start_time = default_timer()
    res = minimize(func.evaluate,
                   x0,
                   method='BFGS',
                   jac=func.derivative,
                   hess=func.hessian,
                   bounds=bounds,
                   callback=optim_history.add_sample,
                   options={'gtol': 1e-6, 'disp': True})
    end_time = default_timer()
    
    print('Optimization time: {} s'.format(end_time - start_time))
    print('Final parameter vector: {}'.format(res.x))
    plot_function_contour_with_samples(function,
                                       optim_history.get_history(),
                                       show_arrows=True,
                                       interesting_points=[(x0, 'r*')])


def global_optimization(function):
    """
    Global optimization of a function
    """
    func = function['function']
    bounds = function['bounds']
    x0 = function['starting_point']

    optim_points = OptimizationAcceptedPointList()

    start_time = default_timer()
    minimizer_kwargs = {"method": "BFGS", "jac": func.derivative}
    res = basinhopping(func.evaluate,
                       x0,
                       minimizer_kwargs=minimizer_kwargs,
                       niter=200,
                       callback=optim_points.basinhopping_callback,
                       disp=None)
    end_time = default_timer()

    optimal_point = res.x
    nearest_optimal_point = optim_points.nearest_point(x0)
    farthest_optimal_point = optim_points.farthest_point(x0)
    delta_optimal_point = optim_points.most_delta_change_point(x0)
    proportional_optimal_point = optim_points.most_proportional_change_point(x0)

    interesting_points = [
        (x0, 'r*'),                        # Starting point with a red start
        (optimal_point, 'g*'),             # Global optimum with a green start
        (nearest_optimal_point, 'mP'),     # Nearest optimum with a magenta +
        (farthest_optimal_point, 'mH'),    # Farthest optimum with a magenta hexagon
        (delta_optimal_point, 'ys'),       # Most delta change optimum with a yellow square
        (proportional_optimal_point, 'y8') # Most proportional change optimum with a yellow octagon
    ]
    
    print('Optimization time: {} s'.format(end_time - start_time))
    print('Basinhopping final parameter vector: {}'.format(optimal_point))
    print('Is solution unique: {}'.format(optim_points.is_unique_point()))
    print('Solution nearest to the starting point: {}'.format(nearest_optimal_point))
    print('Solution farthest to the starting point: {}'.format(farthest_optimal_point))
    print('Solution with the most delta change: {}'.format(delta_optimal_point))
    print('Solution with the most proportional change: {}'.format(proportional_optimal_point))
    plot_function_contour_with_samples(function,
                                       optim_points.get_points(),
                                       show_arrows=False,
                                       interesting_points=interesting_points)


def main():
    functions = generate_functions()
    # Optimization of the function
    optimization(functions['rosen'])
    global_optimization(functions['rosen'])
    global_optimization(functions['underdetermined_circle'])
    global_optimization(functions['underdetermined_disk'])

if __name__ == "__main__":
    main()
