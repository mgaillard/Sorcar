import bpy

import numpy as np
from scipy.optimize import minimize
from ..debug import log

class ScInverseModelingSolver:

    def __init__(self, curr_tree, target_bounding_boxes, initial_float_properties):
        self.curr_tree = curr_tree
        self.target_bounding_boxes = target_bounding_boxes
        self.initial_float_properties = initial_float_properties
        self.property_map = self.create_properties_map(self.initial_float_properties)
        self.cost_function = None


    @staticmethod
    def create_properties_map(float_properties):
        property_map = []
        for property_name in float_properties:
            property_map.append(property_name)
        return property_map


    @staticmethod
    def properties_to_flat_vector(property_map, float_properties):
        flat_vector = []
        for i in range(len(property_map)):
            property_name = property_map[i]
            if property_name in float_properties:
                flat_vector.append(float_properties[property_name])
            else:
                flat_vector.append(0.0)
        return np.array(flat_vector)
    

    @staticmethod
    def flat_vector_to_properties(property_map, flat_vector):
        float_properties = {}
        for i in range(len(property_map)):
            property_name = property_map[i]
            if i < len(flat_vector):
                float_properties[property_name] = flat_vector[i]
            else:
                float_properties[property_name] = 0.0
        return float_properties

    
    def evaluate_cost_function(self, x):
        error = 0.0
        gradient = 0.0
        if self.cost_function is not None:
            # Just a reference for more concise code
            autodiff_variables = self.curr_tree.autodiff_variables
            # Update parameters in the tree directly in the autodiff_variables object
            float_properties = self.flat_vector_to_properties(self.property_map, x)
            for property_name in float_properties:
                if autodiff_variables.has_variable(property_name):
                    autodiff_variables.set_variable_value(property_name, float_properties[property_name])
            # Evaluate the cost and the gradient
            error = autodiff_variables.evaluate_value(self.cost_function)
            gradient_properties = autodiff_variables.evaluate_gradient(self.cost_function)
            gradient = self.properties_to_flat_vector(self.property_map, gradient_properties)
            log("ScInverseModelingSolver", None, "evaluate_cost_function", "f:{}, g:{}".format(repr(error), repr(gradient)), level=1)
        return (error, gradient)
    
    def solve(self):
        # Execute the graph with the initial parameters
        self.curr_tree.set_float_properties(self.initial_float_properties)
        # Collect the name of autodiff bounding boxes
        bounding_boxes = self.curr_tree.get_object_autodiff_boxes_names()
        # TODO: check that all bounding boxes are autodiff types,
        #       if yes, use the autodiff solver, otherwise, use the traditional solver
        # Build the cost function and save it for later
        self.cost_function = self.curr_tree.autodiff_variables.build_cost_function(self.target_bounding_boxes, bounding_boxes, self.curr_tree.objects)
        x0 = self.properties_to_flat_vector(self.property_map, self.initial_float_properties)
        # TODO: use the "L-BFGS-B" solver and define bounds
        res = minimize(self.evaluate_cost_function, x0, method='BFGS', jac=True, options={'gtol': 1e-6, 'disp': True})
        return self.flat_vector_to_properties(self.property_map, res.x)

        