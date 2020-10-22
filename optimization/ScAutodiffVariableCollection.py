import numpy as np

from sys import path
path.append(r"casadi-py37-v3.5.5")
import casadi

class ScAutodiffVariable:

    # TODO: Is it an input or an intermediary value?

    def __init__(self, name, value):
        self.value = value
        self.autodiff = casadi.MX.sym(name, 1)

    def get_symbol(self):
        return self.autodiff

    def set_symbol(self, symbol):
        self.autodiff = symbol

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value


class ScAutodiffVariableCollection:

    def __init__(self):
        self.variables = {}

    def clear(self):
        self.variables.clear()

    def has(self, name):
        return name in self.variables

    def get_variable(self, name):
        if name in self.variables:
            return self.variables[name].get_symbol()
        else:
            return None

    def set_variable(self, name, symbol, value):
        if name in self.variables:
            self.variables[name].set_value(value)
        else:
            self.variables[name] = ScAutodiffVariable(name, value)
        self.variables[name].set_symbol(symbol)

    def get_value(self, name, default_value):
        if name in self.variables:
            return self.variables[name].get_value()
        else:
            return default_value

    def set_value(self, name, value):
        if name in self.variables:
            self.variables[name].set_value(value)
        else:
            self.variables[name] = ScAutodiffVariable(name, value)

    def get_symbols_values(self, symbols):
        """ Return values of symbols """
        values = []
        for symbol in symbols:
            symbol_name = symbol.name()
            if symbol_name in self.variables:
                values.append(self.variables[symbol_name].get_value())
        return values

    def evaluate_value(self, variable):
        """ Evaluate the value of a variable """
        # List symbols in variable and assign them a value
        symbols = casadi.symvar(variable)
        values = self.get_symbols_values(symbols)
        # Build the function
        f = casadi.Function('f', symbols, [variable])
        # Evaluate the function with the values
        result = f.call(values)
        # Convert the output
        return float(result[0])
    
    def evaluate_derivative(self, variable, derivative_name):
        """ Evaluate the derivative of a variable according to another variable """
        # Build the gradient
        derivative_variable = self.variables[derivative_name].get_symbol()
        grad = casadi.gradient(variable, derivative_variable)
        # List symbols in variable and assign them a value
        symbols = casadi.symvar(variable)
        values = self.get_symbols_values(symbols)
        # Build the gradient function
        f = casadi.Function('f', symbols, [grad])
        # Evaluate the gradient with the values
        result = f.call(values)
        # Convert the output
        return float(result[0])

