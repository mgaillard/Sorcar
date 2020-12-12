import mathutils

import numpy as np

from .ScOrientedBoundingBox import ScOrientedBoundingBox

from sys import path
path.append(r"casadi-py37-v3.5.5")
import casadi

class ScAutodiffVariable:

    def __init__(self, name, constant, bounded, minimum, maximum, value):
        self.constant = constant
        self.bounded = bounded
        self.minimum = minimum
        self.maximum = maximum
        self.value = value
        # Difference between a constant and a symbol in casadi
        if not self.constant:
            self.autodiff = casadi.MX.sym(name)
        else:
            self.autodiff = casadi.MX([value])

    def get_constness(self):
        return self.constant

    def get_symbol(self):
        return self.autodiff

    def set_symbol(self, symbol):
        self.autodiff = symbol

    def set_value(self, value):
        self.value = value
        # Clamp according to the minimum and maximum values
        if self.bounded:
            self.value = max(self.value, self.minimum)
            self.value = min(self.value, self.maximum)
        if self.constant:
             self.autodiff = casadi.MX([value])

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
    def fromCenterAndExtent(cls, center_x, extent):
        # By default the center is the origin
        center = casadi.MX.zeros(3, 1)
        center[0] = center_x
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

    @classmethod
    def fromOrientedBoundingBoxAndAxisSystem(cls, box, axis_system):
        # Convert to homogeneous coordinates
        box_center_homogeneous = casadi.vertcat(box.center, casadi.MX([1.0]))
        # Transform in homogeneous space
        center_homogeneous = casadi.mtimes(axis_system.matrix, box_center_homogeneous)
        # Transform back to 3D
        center = casadi.MX(3, 1)
        center[0] = center_homogeneous[0] / center_homogeneous[3]
        center[1] = center_homogeneous[1] / center_homogeneous[3]
        center[2] = center_homogeneous[2] / center_homogeneous[3]
        
        # Transform axis of the box
        axis = [
            casadi.mtimes(axis_system.matrix[0:3, 0:3], box.axis[0]),
            casadi.mtimes(axis_system.matrix[0:3, 0:3], box.axis[1]),
            casadi.mtimes(axis_system.matrix[0:3, 0:3], box.axis[2])
        ]

        extent = casadi.MX(3, 1)
        extent[0] = box.extent[0] * casadi.norm_2(axis[0])
        extent[1] = box.extent[1] * casadi.norm_2(axis[1])
        extent[2] = box.extent[2] * casadi.norm_2(axis[2])
        
        # Normalize axis vectors
        axis[0] = axis[0] / casadi.norm_2(axis[0])
        axis[1] = axis[1] / casadi.norm_2(axis[1])
        axis[2] = axis[2] / casadi.norm_2(axis[2])
        
        return cls(center, axis, extent)

    def set_center_x(self, center_x):
        self.center[0] = center_x
        
    def set_center_y(self, center_y):
        self.center[1] = center_y

    def set_center_z(self, center_z):
        self.center[2] = center_z

    def set_center(self, center):
        self.center = center

    def get_center(self):
        return self.center

    def reset_rotation(self):
        self.axis = [
            casadi.MX([1.0, 0.0, 0.0]),
            casadi.MX([0.0, 1.0, 0.0]),
            casadi.MX([0.0, 0.0, 1.0])
        ]

    def rotate_x(self, angle_x):
        cx = casadi.cos(angle_x)
        sx = casadi.sin(angle_x)

        # Rotate box X axis
        axis_01 = cx * self.axis[0][1] - sx * self.axis[0][2]
        axis_02 = sx * self.axis[0][1] + cx * self.axis[0][2]
        # Rotate box Y axis
        axis_11 = cx * self.axis[1][1] - sx * self.axis[1][2]
        axis_12 = sx * self.axis[1][1] + cx * self.axis[1][2]
        # Rotate box Z axis
        axis_21 = cx * self.axis[2][1] - sx * self.axis[2][2]
        axis_22 = sx * self.axis[2][1] + cx * self.axis[2][2]

        # Assign new values
        self.axis[0][1] = axis_01
        self.axis[0][2] = axis_02
        self.axis[1][1] = axis_11
        self.axis[1][2] = axis_12
        self.axis[2][1] = axis_21
        self.axis[2][2] = axis_22

    def rotate_y(self, angle_y):
        cy = casadi.cos(angle_y)
        sy = casadi.sin(angle_y)

        # Rotate box X axis
        axis_00 =  cy * self.axis[0][0] + sy * self.axis[0][2]
        axis_02 = -sy * self.axis[0][0] + cy * self.axis[0][2]
        # Rotate box Y axis
        axis_10 =  cy * self.axis[1][0] + sy * self.axis[1][2]
        axis_12 = -sy * self.axis[1][0] + cy * self.axis[1][2]
        # Rotate box Z axis
        axis_20 =  cy * self.axis[2][0] + sy * self.axis[2][2]
        axis_22 = -sy * self.axis[2][0] + cy * self.axis[2][2]

        # Assign new values
        self.axis[0][0] = axis_00
        self.axis[0][2] = axis_02
        self.axis[1][0] = axis_10
        self.axis[1][2] = axis_12
        self.axis[2][0] = axis_20
        self.axis[2][2] = axis_22

    def rotate_z(self, angle_z):
        cz = casadi.cos(angle_z)
        sz = casadi.sin(angle_z)

        # Rotate box X axis
        axis_00 = cz * self.axis[0][0] - sz * self.axis[0][1]
        axis_01 = sz * self.axis[0][0] + cz * self.axis[0][1]
        # Rotate box Y axis
        axis_10 = cz * self.axis[1][0] - sz * self.axis[1][1]
        axis_11 = sz * self.axis[1][0] + cz * self.axis[1][1]
        # Rotate box Z axis
        axis_20 = cz * self.axis[2][0] - sz * self.axis[2][1]
        axis_21 = sz * self.axis[2][0] + cz * self.axis[2][1]

        # Assign new values
        self.axis[0][0] = axis_00
        self.axis[0][1] = axis_01
        self.axis[1][0] = axis_10
        self.axis[1][1] = axis_11
        self.axis[2][0] = axis_20
        self.axis[2][1] = axis_21

    def set_rotation(self, angle_x, angle_y, angle_z):
        # Reset rotation
        self.reset_rotation()
        # Rotate each angle in order 'XYZ'
        self.rotate_x(angle_x)
        self.rotate_y(angle_y)
        self.rotate_z(angle_z)

    def get_axis(self, index):
        return self.axis[index]

    def set_extent_x(self, extent_x):
        self.extent[0] = extent_x
        
    def set_extent_y(self, extent_y):
        self.extent[1] = extent_y

    def set_extent_z(self, extent_z):
        self.extent[2] = extent_z
    
    def set_extent(self, extent):
        self.extent = extent

    def get_extent_x(self):
        return self.extent[0]

    def get_extent_y(self):
        return self.extent[1]

    def get_extent_z(self):
        return self.extent[2]

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

    def list_box_edges(self, offset=0):
        return [
            (offset + 0, offset + 1), (offset + 0, offset + 2), (offset + 0, offset + 3),
            (offset + 1, offset + 4), (offset + 1, offset + 5), (offset + 2, offset + 4),
            (offset + 2, offset + 6), (offset + 3, offset + 5), (offset + 3, offset + 6),
            (offset + 4, offset + 7), (offset + 5, offset + 7), (offset + 6, offset + 7)
        ]

    def list_box_faces(self, offset=0):
        return [
            (offset + 0, offset + 2, offset + 4, offset + 1),
            (offset + 0, offset + 3, offset + 6, offset + 2),
            (offset + 1, offset + 4, offset + 7, offset + 5), 
            (offset + 3, offset + 5, offset + 7, offset + 6),
            (offset + 0, offset + 1, offset + 5, offset + 3),
            (offset + 2, offset + 6, offset + 7, offset + 4)
        ]


class ScAutodiffAxisSystem:

    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def fromDefault(cls):
        return cls(casadi.MX.eye(4))

    @classmethod
    def compose(cls, parent_axis_system, child_axis_system):
        """ Compose this axis system with another axis system """
        return cls(casadi.mtimes(parent_axis_system.matrix, child_axis_system.matrix))

    def set_translation_x(self, translation_x):
        self.matrix[0, 3] = translation_x
        
    def set_translation_y(self, translation_y):
        self.matrix[1, 3] = translation_y

    def set_translation_z(self, translation_z):
        self.matrix[2, 3] = translation_z

    def set_translation(self, translation_x, translation_y, translation_z):
        self.set_translation_x(translation_x)
        self.set_translation_y(translation_y)
        self.set_translation_z(translation_z)

    def translate_x(self, translation_x):
        self.matrix[0, 3] = self.matrix[0, 3] + translation_x
        
    def translate_y(self, translation_y):
        self.matrix[1, 3] = self.matrix[1, 3] + translation_y

    def translate_z(self, translation_z):
        self.matrix[2, 3] = self.matrix[2, 3] + translation_z

    def translate(self, translation_x, translation_y, translation_z):
        self.translate_x(translation_x)
        self.translate_y(translation_y)
        self.translate_z(translation_z)

    def set_scale_x(self, scale_x):
        self.matrix[0, 0] = scale_x
        
    def set_scale_y(self, scale_y):
        self.matrix[1, 1] = scale_y

    def set_scale_z(self, scale_z):
        self.matrix[2, 2] = scale_z

    def set_scale(self, scale_x, scale_y, scale_z):
        self.set_scale_x(scale_x)
        self.set_scale_y(scale_y)
        self.set_scale_z(scale_z)

    def scale(self, scale_x, scale_y, scale_z):
         # Scale matrix
        matrix_scale = casadi.MX.eye(4)
        matrix_scale[0, 0] = scale_x
        matrix_scale[1, 1] = scale_y
        matrix_scale[2, 2] = scale_z

        self.matrix = casadi.mtimes(matrix_scale, self.matrix)

    def reset_rotation(self):
        self.matrix[0:3, 0:3] = casadi.MX.eye(3)

    def rotate_x(self, angle_x):
        cx = casadi.cos(angle_x)
        sx = casadi.sin(angle_x)

        # Rotation matrix around X
        rotation_x = casadi.MX.eye(3)
        rotation_x[1, 1] = cx
        rotation_x[1, 2] = -sx
        rotation_x[2, 1] = sx
        rotation_x[2, 2] = cx

        self.matrix[0:3, 0:3] = casadi.mtimes(rotation_x, self.matrix[0:3, 0:3])

    def rotate_y(self, angle_y):
        cy = casadi.cos(angle_y)
        sy = casadi.sin(angle_y)

        # Rotation matrix around X
        rotation_y = casadi.MX.eye(3)
        rotation_y[0, 0] = cy
        rotation_y[0, 2] = sy
        rotation_y[2, 0] = -sy
        rotation_y[2, 2] = cy

        self.matrix[0:3, 0:3] = casadi.mtimes(rotation_y, self.matrix[0:3, 0:3])

    def rotate_z(self, angle_z):
        cz = casadi.cos(angle_z)
        sz = casadi.sin(angle_z)

        # Rotation matrix around X
        rotation_z = casadi.MX.eye(3)
        rotation_z[0, 0] = cz
        rotation_z[0, 1] = -sz
        rotation_z[1, 0] = sz
        rotation_z[1, 1] = cz

        self.matrix[0:3, 0:3] = casadi.mtimes(rotation_z, self.matrix[0:3, 0:3])

    def rotate(self, angle_x, angle_y, angle_z):
        # Rotate each angle in order 'XYZ'
        self.rotate_x(angle_x)
        self.rotate_y(angle_y)
        self.rotate_z(angle_z)

    def set_rotation(self, angle_x, angle_y, angle_z):
        # Reset rotation
        self.reset_rotation()
        # Rotate each angle in order 'XYZ'
        self.rotate(angle_x, angle_y, angle_z)


class ScAutodiffVariableCollection:

    def __init__(self):
        self.variables = {}
        self.axis_systems = {}
        self.boxes = {}

    def clear(self):
        self.variables.clear()
        self.boxes.clear()

    # --- Function for accessing variables ---

    def has_variable(self, name):
        return name in self.variables

    def create_variable(self, name, constant, bounded, minimum, maximum, value):
        self.variables[name] = ScAutodiffVariable(name, constant, bounded, minimum, maximum, value)

    def set_variable_symbol(self, name, symbol, value):
        if name in self.variables:
            self.variables[name].set_value(value)
            self.variables[name].set_symbol(symbol)

    def get_variable_symbol(self, name):
        if name in self.variables:
            return self.variables[name].get_symbol()
        else:
            return None

    def get_variable_value(self, name, default_value):
        if name in self.variables:
            return self.variables[name].get_value()
        else:
            return default_value

    def set_variable_value(self, name, value):
        if name in self.variables:
            self.variables[name].set_value(value)

    def get_variable_constness(self, name):
        if name in self.variables:
            return self.variables[name].get_constness()
        else:
            return False

    def get_temporary_const_variable(self, value):
        return casadi.MX([value])

    # --- Function for accessing axis systems ---

    def has_axis_system(self, name):
        return name in self.axis_systems

    def get_axis_system(self, name):
        if name in self.axis_systems:
            return self.axis_systems[name]
        else:
            return None

    def create_default_axis_system(self, name):
        self.axis_systems[name] = ScAutodiffAxisSystem.fromDefault()
        return self.axis_systems[name]

    # --- Function for accessing boxes ---

    def has_box(self, name):
        return name in self.boxes

    def get_box(self, name):
        if name in self.boxes:
            return self.boxes[name]
        else:
            return None

    def set_box_extent(self, name, extent):
        if name in self.boxes:
            self.boxes[name].set_extent(extent)
        else:
            self.boxes[name] = ScAutodiffOrientedBoundingBox.fromExtent(extent)

    def set_box_from_constants(self, name, box):
        self.boxes[name] = ScAutodiffOrientedBoundingBox.fromConstantOrientedBoundingBox(box)

    # --- Function for optimization ---

    def get_parent_names(self, objects, object_name):
        """ Give the list of names of parents of an object in reverse order """
        # Find the object by name
        obj = None
        for i in range(len(objects)):
            if objects[i].name == object_name:
                obj = objects[i]

        parent_names = []
        
        # If object was found
        if obj is not None:
            obj = obj.parent

        # Iterate over parents
        while obj is not None:
            parent_names.append(obj.name)
            # Get the parent
            obj = obj.parent
        
        # Reverse the list of parents
        parent_names.reverse()

        return parent_names

    def compose_hierarchy_axis_system(self, parent_names):
        """ Compose axis systems according to axis system names """
        axis_system = ScAutodiffAxisSystem.fromDefault()

        for parent_name in parent_names:
            current_axis_system = self.get_axis_system(parent_name)
            if current_axis_system is not None:
                axis_system = ScAutodiffAxisSystem.compose(axis_system, current_axis_system)

        return axis_system

    def compute_transformed_bounding_box(self, objects, object_name):
        # Get the object bounding box and transform it according to its parents
        object_box = self.boxes[object_name]
        # Get parents of object in reverse order
        hierarchy_object_names = self.get_parent_names(objects, object_name)
        # Add the object to the list of names
        hierarchy_object_names.append(object_name)
        # Get the total transformation for the current object
        axis_system = self.compose_hierarchy_axis_system(hierarchy_object_names)
        return ScAutodiffOrientedBoundingBox.fromOrientedBoundingBoxAndAxisSystem(object_box, axis_system)

    def build_cost_function(self, target_bounding_boxes, bounding_boxes, objects):
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
                # Get the object bounding box and transform it according to its parents
                transformed_bounding_box = self.compute_transformed_bounding_box(objects, object_name)
                # List points to match between the two objects
                target_box_points = target_autodiff_boxes[object_name].list_points_to_match()
                box_points = transformed_bounding_box.list_points_to_match()
                number_points = min(len(target_box_points), len(box_points))
                # The error is the sum of square distances between corners of the bounding boxes
                for i in range(number_points):
                    error = error + (target_box_points[i][0] - box_points[i][0])**2
                    error = error + (target_box_points[i][1] - box_points[i][1])**2
                    error = error + (target_box_points[i][2] - box_points[i][2])**2
        
        return error

    def keep_only_free_symbols(self, symbols):
        """ Return the list of non constant symbols """
        free_symbols = []
        for symbol in symbols:
            symbol_name = symbol.name()
            if not self.get_variable_constness(symbol_name):
                free_symbols.append(symbol)
        return free_symbols

    def get_symbols_values(self, symbols):
        """ Return values of symbols """
        values = []
        for symbol in symbols:
            symbol_name = symbol.name()
            if self.has_variable(symbol_name):
                values.append(self.variables[symbol_name].get_value())
        return values

    def evaluate(self, variable):
        # List symbols in variable and assign them a value
        symbols = casadi.symvar(variable)
        values = self.get_symbols_values(symbols)
        # Build the function
        f = casadi.Function('f', symbols, [variable])
        # Evaluate the function with the values
        return f.call(values)

    def evaluate_value(self, variable):
        """ Evaluate the value of a variable return a single float """
        result = self.evaluate(variable)
        # Convert the output
        return float(result[0])

    def evaluate_vector(self, variable):
        """ Evaluate the value of a variable return a list of float """
        results = self.evaluate(variable)
        # Convert values in the results array to float
        results_float = []
        for i in range(results[0].size1()):
            results_float.append(float(results[0][i]))
        return results_float

    def evaluate_matrix(self, variable):
        """ Evaluate the value of a variable return a matrix of float """
        results = self.evaluate(variable)
        # Convert values in the results array to float
        results_float = []
        for i in range(results[0].size1()):
            row = []
            for j in range(results[0].size2()):
                row.append(float(results[0][i, j]))
            results_float.append(row)
        return results_float
    
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
        # List symbols in variable 
        symbols = casadi.symvar(variable)
        # Only keep free variables (not the constants)
        free_symbols = self.keep_only_free_symbols(symbols)
        # Build the gradient with free variables
        grad = casadi.gradient(variable, casadi.vertcat(*free_symbols))
        # Assign symbols a value
        values = self.get_symbols_values(symbols)
        # Build the gradient function
        f = casadi.Function('f', symbols, [grad])
        # Evaluate the gradient with the values
        results = f.call(values)
        # Convert values in the results array to float
        output = {}
        for i in range(len(free_symbols)):
            variable_name = free_symbols[i].name()
            variable_value = results[0][i]
            output[variable_name] = float(variable_value)
        return output