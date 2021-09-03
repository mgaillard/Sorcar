import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
import matplotlib.pyplot as plt

# TODO: target function to optimize
# TODO: optimize with SciPy
# TODO: measure optimization time
# TODO: record path of the optimization
# TODO: plot the 2D function 
# TODO: plot the optimization path
# TODO: implement the benchmark function with CasADi and use AD for Jacobian and Hessian
# TODO: implement a list for function optimization, and minize all of them

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


def plot_function_contour():
    x = np.linspace(-2.0, 2.0, 50)
    y = np.linspace(-1.0, 3.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(X, Y, Z, 20)

    plt.show()


def optimization():
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = minimize(function, x0, method='BFGS', jac=function_der, options={'gtol': 1e-6, 'disp': True})
    print(res.x)


def main():
    optimization()
    plot_function_contour()


if __name__ == "__main__":
    main()
