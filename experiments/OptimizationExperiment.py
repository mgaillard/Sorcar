from timeit import default_timer
import numpy as np
from casadi import *
from scipy.optimize import minimize, rosen, rosen_der
import matplotlib.pyplot as plt

# TODO: animation of the plot
# TODO: implement the Hessian in the Casadi function
# TODO: implement a list for function optimization, and minize all of them with different optimizers

class OptimizationHistory:
    """ Register evaluations of the cost function as optimization is happening """

    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def get_history(self):
        return np.array(self.history)

    def callback(self, xk):
        """
        Callback for scipy.optimize.minimize()
        Parameter xk is the current parameter vector 
        """
        # Note that we make a deep copy of xk
        self.history.append(np.copy(xk))


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


def plot_function_surface():
    x = np.linspace(-2.0, 2.0, 50)
    y = np.linspace(-1.0, 3.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z)

    plt.show()


def plot_function_contour_with_samples(function, path):
    """
    2D contour plot of the function with optimization path
    Source: http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
    """
    transpose_path = path.T

    resolutionX = 50
    resolutionY = 50

    x = np.linspace(-2.0, 2.0, resolutionX)
    y = np.linspace(-1.0, 3.0, resolutionY)
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

    start_time = default_timer()
    res = minimize(function.evaluate,
                   x0,
                   method='BFGS',
                   jac=function.derivative,
                   callback=optim_history.callback,
                   options={'gtol': 1e-6, 'disp': True})
    end_time = default_timer()
    
    print('Optimization time: {} s'.format(end_time - start_time))
    print('Final parameter vector: {}'.format(res.x))
    plot_function_contour_with_samples(function, optim_history.get_history())
    


def main():
    # Rosenbrock function in CasADi
    x = MX.sym('x', 2)
    expr = (1.0 - x[0])*(1.0 - x[0]) + 100.0*(x[1] - x[0]*x[0])*(x[1] - x[0]*x[0])
    function = CasadiFunction(expr, x)
    # Optimization of the function
    optimization(function)


if __name__ == "__main__":
    main()
