import mathutils

import numpy as np

from .ScOrientedBoundingBox import ScOrientedBoundingBox

from sys import path
path.append(r"casadi-py37-v3.5.5")
import casadi

class ScAutodiffVariable:

    def __init__(self, name, value):
        # TODO: add a minimum and maximum value
        self.value = value
        self.autodiff = casadi.MX.sym(name)

    def get_symbol(self):
        return self.autodiff

    def set_symbol(self, symbol):
        self.autodiff = symbol

    def set_value(self, value):
        # TODO: clamp according to the minimum and maximum values
        self.value = value

    def get_value(self):
        return self.value


class ScAutodiffOrientedBoundingBox:

    def __init__(self, center, axis, extent):
        self.center = center
        self.axis = axis
        self.extent = extent
    
    @classmethod
    def fromExtent(cls, extent):
        # By default the center is the origin
        center = casadi.MX.zeros(3, 1)
        # By default the axis are X, Y and Z
        axis = [
            casadi.MX([1.0, 0.0, 0.0]),
            casadi.MX([0.0, 1.0, 0.0]),
            casadi.MX([0.0, 0.0, 1.0])
        ]
        return cls(center, axis, extent)

    @classmethod
    def fromConstantOrientedBoundingBox(cls, box):
        center = casadi.MX([box.center[0], box.center[1], box.center[2]])
        axis = [
            casadi.MX([box.axis[0][0], box.axis[0][1], box.axis[0][2]]),
            casadi.MX([box.axis[1][0], box.axis[1][1], box.axis[1][2]]),
            casadi.MX([box.axis[2][0], box.axis[2][1], box.axis[2][2]])
        ]
        extent = casadi.MX([box.extent[0], box.extent[1], box.extent[2]])
        return cls(center, axis, extent)

    def set_center(self, center):
        self.center = center

    def get_center(self):
        return self.center

    def get_axis(self, index):
        return self.axis[index]

    def set_extent(self, extent):
        self.extent = extent

    def get_extent(self):
        return self.extent

    def list_points_to_match(self):
        return [
            self.center - self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center - self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center - self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] - self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] - self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2],
            self.center - self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2],
            self.center + self.extent[0] * self.axis[0] + self.extent[1] * self.axis[1] + self.extent[2] * self.axis[2]
        ]


class ScAutodiffVariableCollection:

    def __init__(self):
        self.variables = {}
        self.boxes = {}

    def clear(self):
        self.variables.clear()
        self.boxes.clear()

    def has_variable(self, name):
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

    def has_box(self, name):
        return name in self.boxes

    def set_box(self, name, extent):
        if name in self.boxes:
            self.boxes[name].set_extent(extent)
        else:
            self.boxes[name] = ScAutodiffOrientedBoundingBox.fromExtent(extent)

    def get_box(self, name):
        if name in self.boxes:
            # Convert the autodiff bounding box to a regular box
            box = self.boxes[name]

            # Convert center to a Blender vector
            center = mathutils.Vector((
                self.evaluate_value(box.get_center()[0]),
                self.evaluate_value(box.get_center()[1]),
                self.evaluate_value(box.get_center()[2])
            ))
            
            # Convert axis to a Blender vectors
            axis = []
            for i in range(3):
                axis.append(mathutils.Vector((
                    self.evaluate_value(box.axis[i][0]),
                    self.evaluate_value(box.axis[i][1]),
                    self.evaluate_value(box.axis[i][2])
                )))

            # Convert extent to a Blender vector
            extent = mathutils.Vector((
                self.evaluate_value(box.get_extent()[0]),
                self.evaluate_value(box.get_extent()[1]),
                self.evaluate_value(box.get_extent()[2]),
            ))

            return ScOrientedBoundingBox(center, axis, extent)

    def build_cost_function(self, target_bounding_boxes, bounding_boxes):
        """ Build a cost function according to a list of target bounding boxes """
        # Convert the target boxes to the autodiff bouding box format
        target_autodiff_boxes = {}
        for box_name in target_bounding_boxes:
            box = target_bounding_boxes[box_name]
            target_autodiff_boxes[box_name] = ScAutodiffOrientedBoundingBox.fromConstantOrientedBoundingBox(box)
        # The error is a symbolic function
        error = casadi.MX.zeros(1, 1)
        # For each bounding_boxes compare the to the target
        for object_name in bounding_boxes:
            # Find the corresponding box in the target
            if (object_name in self.boxes) and (object_name in target_autodiff_boxes):
                # List points to match between the two objects
                target_box_points = target_autodiff_boxes[object_name].list_points_to_match()
                box_points = self.boxes[object_name].list_points_to_match()
                number_points = min(len(target_box_points), len(box_points))
                # The error is the sum of square distances between corners of the bounding boxes
                for i in range(number_points):
                    error = error + (target_box_points[i][0] - box_points[i][0])**2
                    error = error + (target_box_points[i][1] - box_points[i][1])**2
                    error = error + (target_box_points[i][2] - box_points[i][2])**2
        
        return error

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
        # TODO: make this work for many derivatives, to compute a gradient
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

    def evaluate_gradient(self, variable):
        """ Evaluate the gradient of a variable """
        # TODO: check that it works for two variables and check that the order is the same as the function that calls this function
        # List symbols in variable 
        symbols = casadi.symvar(variable)
        # Build the gradient
        grad = casadi.gradient(variable, casadi.vertcat(*symbols))
        # Assign symbols a value
        values = self.get_symbols_values(symbols)
        # Build the gradient function
        f = casadi.Function('f', symbols, [grad])
        # Evaluate the gradient with the values
        results = f.call(values)
        # Convert values in the results array to float
        return [float(v) for v in results]
        