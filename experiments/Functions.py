import math
from casadi import *

class CasadiFunction:
    """
    Casadi wrapper for a function to optimize.
    Jacobian and Hessian matrices are computed using automatic differentiation
    """

    def __init__(self, activate_hessian=True):
        """
        Initialize the function to optimize and its gradient
        Parameter activate_hessian is a boolean to activate computation/loading of the Hessian
        """
        self.dimensionality = 0
        self.func = None
        self.grad_func = None
        self.hess_func = None
        self.activate_hessian = activate_hessian

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
        # Only compute the Hessian if it is activated
        if self.activate_hessian:
            # Build CasADi Function for evaluating the Hessian of the function to optimize
            hess_expression, g = hessian(expression, symbols)
            self.hess_func = Function('h', [symbols], [hess_expression], function_options)
        else:
            self.hess_func = None

    def set_functions(self, func, grad_func, hess_func=None):
        """
        Set the function to optimize.
        The function is directly given, no need to build the gradient nor the Hessian.
        """
        # Directly set the function
        self.func = func
        # Infer the dimensionality from the function input size
        self.dimensionality = self.func.size1_in(0)
        # Directly set the gradient function
        self.grad_func = grad_func
        # Directly set the Hessian function
        self.hess_func = hess_func
        # If the Hessian function was not given, deactivate it
        if self.hess_func:
            self.activate_hessian = False

    def evaluate(self, x):
        # Check that the function is defined
        # Check that the input has the right dimensionality
        if (self.func is not None) and (len(x) == self.dimensionality):
            # Call the function with the parameters
            result = self.func.call([x])
            # Convert the output to float
            return float(result[0])
        # By default, the output is None
        return None

    def derivative(self, x):
        # Check that the gradient function is defined
        # Check that the input has the right dimensionality
        if (self.grad_func is not None) and (len(x) == self.dimensionality):
            # Call the function with the parameters
            results = self.grad_func.call([x])
            # Convert the output to float
            output = []
            for i in range(self.dimensionality):
                output.append(float(results[0][i]))
            return output
        # By default, the output is None
        return None

    def is_hessian_active(self):
        """ Return true if the Hessian is activated, false otherwise """
        return (self.hess_func is not None)

    def hessian(self, x):
        # Check that the Hessian function is defined
        # Check that the input has the right dimensionality
        if (self.hess_func is not None) and (len(x) == self.dimensionality):
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

        # Check that all functions are defined
        # Check that both inputs have the right dimensionality
        if ((self.func is not None)
        and (self.grad_func is not None)
        and (self.hess_func is not None)
        and (len(x) == self.dimensionality)
        and (len(p) == self.dimensionality)):

            dx = np.array(p) - np.array(x)
            f = np.array(self.evaluate(x))
            g = np.array(self.derivative(x))
            H = np.array(self.hessian(x))

            approximation = f + np.dot(g, dx) + 0.5 * np.dot(dx, np.matmul(H, dx))

            return approximation
        else:
            return None

    def generate_code(self, filename):
        """
        Generate C code associated to the functions for later JIT compilation
        """
        generator = CodeGenerator(filename)
        if self.func is not None:
            generator.add(self.func)
        if self.grad_func is not None:
            generator.add(self.grad_func)
        if self.hess_func is not None:
            generator.add(self.hess_func)
        generator.generate()

    def load_and_jit_compile_code(self, filename):
        """
        Load a previously saved function
        JIT compile it using the embedded clang in Casadi
        """
        importer = Importer(filename, 'clang')
        self.func = external('f', importer)
        # Infer the dimensionality from the function input size
        self.dimensionality = self.func.size1_in(0)
        self.grad_func = external('g', importer)
        # Only load the Hessian if it is activated
        if self.activate_hessian:
            self.hess_func = external('h', importer)
        else:
            self.hess_func = None

    def load_binary_code(self, filename):
        """
        Load a previously saved function already compiled
        """
        self.func = external('f', filename)
        # Infer the dimensionality from the function input size
        self.dimensionality = self.func.size1_in(0)
        self.grad_func = external('g', filename)
        # Only load the Hessian if it is activated
        if self.activate_hessian:
            self.hess_func = external('h', filename)
        else:
            self.hess_func = None


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
        'starting_point': [1.5, 0.5]
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

    # Compute the eigen vectors
    eigenvalues, eigenvectors = linalg.eigh(np.array(func.hessian(point)))

    contour_lines = 20 # Number of contour lines
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.contourf(X, Y, Z1, contour_lines, cmap=plt.cm.jet)
    ax1.plot(point[0], point[1], 'r*', markersize=10)
    ax2.contourf(X, Y, Z2, contour_lines, cmap=plt.cm.jet)
    ax2.plot(point[0], point[1], 'r*', markersize=10)
    ax3.contourf(X, Y, D, contour_lines, cmap=plt.cm.jet)
    ax3.plot(point[0], point[1], 'r*', markersize=10)

    # Draw arrows for eigen values
    ax2.arrow(point[0], point[1], eigenvectors[0][0], eigenvectors[0][1], width=0.003, color="black")
    print('Eigen value for black arrow: {}'.format(eigenvalues[0]))
    ax2.arrow(point[0], point[1], eigenvectors[1][0], eigenvectors[1][1], width=0.003, color="white")
    print('Eigen value for white arrow: {}'.format(eigenvalues[1]))

    plt.show()
