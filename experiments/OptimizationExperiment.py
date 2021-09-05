from timeit import default_timer
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
import matplotlib.pyplot as plt

# TODO: animation of the plot
# TODO: make it possible to change the target function to optimize
# TODO: implement the benchmark function with CasADi and use AD for Jacobian and Hessian
#       make a class that accepts a CasADi expression and evaluate it and derivate it with class members
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


def function(X):
    return rosen(X)


def function_der(X):
    return rosen_der(X)


def plot_function_surface():
    x = np.linspace(-2.0, 2.0, 50)
    y = np.linspace(-1.0, 3.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z)

    plt.show()


def plot_function_contour_with_samples(path):
    """
    2D contour plot of the function with optimization path
    Source: http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
    """
    transpose_path = path.T

    x = np.linspace(-2.0, 2.0, 50)
    y = np.linspace(-1.0, 3.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    
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


def optimization():
    x0 = [1.3, 0.7]
    optim_history = OptimizationHistory()

    start_time = default_timer()
    res = minimize(function,
                   x0,
                   method='BFGS',
                   jac=function_der,
                   callback=optim_history.callback,
                   options={'gtol': 1e-6, 'disp': True})
    end_time = default_timer()
    
    print('Optimization time: {} s'.format(end_time - start_time))
    print('Final parameter vector: {}'.format(res.x))
    plot_function_contour_with_samples(optim_history.get_history())
    


def main():
    optimization()


if __name__ == "__main__":
    main()
