import math
from timeit import default_timer
import numpy as np
from casadi import *
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt

# TODO: animation of the plot
# TODO: implement a list for function optimization, and minimize all of them with different optimizers
# TODO: add undetermined functions with a solution set that is 1D and curved
# TODO: add undetermined functions with a solution set that is 1D and circular
# TODO: add undetermined function with a solution set that is 2D
# TODO: add global optimization objective function
# TODO: add global optimization methods
# TODO: look at the Hessian of the underdetermined function in the valley

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

    def add_sample_basinhopping(self, x, f, accept):
        """
        Callback for scipy.optimize.basinhopping()
        Parameter x is the current parameter vector
        We ignore other parameters
        """
        # Note that we make a deep copy of xk
        self.history.append(np.copy(x))


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


def generate_rosen():
    """
    2D Rosenbrock function
    a = 1.0
    b = 100.0
    Optimimum: [1.0, 1.0]
    """
    x = SX.sym('x', 2)
    expr = pow(1.0 - x[0], 2) + 100.0*pow(x[1] - x[0]*x[0], 2)
    return CasadiFunction(expr, x)


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
    return CasadiFunction(expr, x)


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
    return CasadiFunction(expr, x)


def generate_rastrigin():
    """
    Rastigrin function
    20.0 + (x^2 - 10.0*cos(2*pi*x)) + (y^2 - 10.0*cos(2*pi*y))
    """
    x = SX.sym('x', 2)
    expr = 20.0 + pow(x[0]*x[0] - 10.0*cos(2.0*math.pi*x[0]), 2) + pow(x[1]*x[1] - 10.0*cos(2.0*math.pi*x[1]), 2)
    return CasadiFunction(expr, x)


def generate_functions():
    """ Generate a list of predefined functions to optimize """
    functions = {
        'rosen': generate_rosen(),
        'underdetermined_linear': generate_underdetermined_linear(),
        'himmelblau': generate_himmelblau(),
        'rastrigin': generate_rastrigin()
    }
    return functions


def plot_function_contour_with_samples(function, path):
    """
    2D contour plot of the function with optimization path
    Source: http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
    """
    transpose_path = path.T

    resolutionX = 50
    resolutionY = 50

    # x = np.linspace(-2.0, 2.0, resolutionX)
    # y = np.linspace(-1.0, 3.0, resolutionY)
    x = np.linspace(-5.0, 5.0, resolutionX)
    y = np.linspace(-5.0, 5.0, resolutionY)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((resolutionX, resolutionY))
    for i in range(resolutionX):
        for j in range(resolutionY):
            Z[i, j] = function.evaluate([X[i, j], Y[i, j]])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Contour plot of the function
    ax.contour(X, Y, Z, 20, cmap=plt.cm.jet)
    # Path with arrows
    ax.quiver(transpose_path[0,:-1],
              transpose_path[1,:-1],
              transpose_path[0,1:] - transpose_path[0,:-1],
              transpose_path[1,1:] - transpose_path[1,:-1],
              scale_units='xy',
              angles='xy',
              scale=1,
              color='k')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    plt.show()


def optimization(function):
    x0 = [1.3, 0.7]
    optim_history = OptimizationHistory()
    optim_history.add_sample(x0)

    start_time = default_timer()
    res = minimize(function.evaluate,
                   x0,
                   method='BFGS',
                   jac=function.derivative,
                   hess=function.hessian,
                   callback=optim_history.add_sample,
                   options={'gtol': 1e-6, 'disp': True})
    end_time = default_timer()
    
    print('Optimization time: {} s'.format(end_time - start_time))
    print('Final parameter vector: {}'.format(res.x))
    plot_function_contour_with_samples(function, optim_history.get_history())


def global_optimization(function):
    x0 = [1.3, 0.7]
    optim_history = OptimizationHistory()
    optim_history.add_sample(x0)

    start_time = default_timer()
    minimizer_kwargs = {"method": "BFGS", "jac": function.derivative}
    res = basinhopping(function.evaluate,
                       x0,
                       minimizer_kwargs=minimizer_kwargs,
                       niter=200,
                       callback=optim_history.add_sample_basinhopping)
    end_time = default_timer()
    
    print('Optimization time: {} s'.format(end_time - start_time))
    print('Final parameter vector: {}'.format(res.x))
    plot_function_contour_with_samples(function, optim_history.get_history())


def main():
    functions = generate_functions()    
    # Optimization of the function
    optimization(functions['rosen'])


if __name__ == "__main__":
    main()
