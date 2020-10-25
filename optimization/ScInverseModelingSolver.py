import bpy

import numpy as np
from scipy.optimize import minimize
from ..debug import log

class ScInverseModelingSolver:

    def __init__(self, curr_tree, target_bounding_boxes, initial_float_properties):
        self.curr_tree = curr_tree
        self.target_bounding_boxes = target_bounding_boxes
        self.initial_float_properties = initial_float_properties
        self.cost_function = None


    @staticmethod
    def properties_to_flat_vector(float_properties):
        flat_vector = []
        for property_name in float_properties:
            flat_vector.append(float_properties[property_name])
        return np.array(flat_vector)


    @staticmethod
    def flat_vector_to_properties(float_properties, flat_vector):
        new_float_properties = float_properties.copy()
        index = 0
        for property_name in new_float_properties:
            new_float_properties[property_name] = flat_vector[index]
            index += 1
        return new_float_properties

    
    def evaluate_cost_function(self, x):
        error = 0.0
        gradient = 0.0
        if self.cost_function is not None:
            # Just a reference for more concise code
            autodiff_variables = self.curr_tree.autodiff_variables
            # Update parameters in the tree directly in the autodiff_variables object
            float_properties = self.flat_vector_to_properties(self.initial_float_properties, x)
            for property_name in float_properties:
                if autodiff_variables.has_variable(property_name):
                    autodiff_variables.set_value(property_name, float_properties[property_name])
            # Evaluate the cost and the gradient
            error = autodiff_variables.evaluate_value(self.cost_function)
            gradient = autodiff_variables.evaluate_gradient(self.cost_function)
            log("ScInverseModelingSolver", None, "evaluate_cost_function", "f:{}, g:{}".format(repr(error), repr(gradient)), level=1)
        return (error, gradient)
    
    def solve(self):
        # Execute the graph with the initial parameters
        self.curr_tree.set_float_properties(self.initial_float_properties)
        self.curr_tree.execute_node()
        # Collect the name of autodiff bounding boxes
        bounding_boxes = self.curr_tree.get_object_autodiff_boxes_names()
        # TODO: check that all bounding boxes are autodiff types,
        #       if yes, use the autodiff solver, otherwise, use the traditional solver
        # Build the cost function and save it for later
        self.cost_function = self.curr_tree.autodiff_variables.build_cost_function(self.target_bounding_boxes,
                                                                                   bounding_boxes)
        x0 = self.properties_to_flat_vector(self.initial_float_properties)
        # TODO: use the "L-BFGS-B" solver and define bounds
        res = minimize(self.evaluate_cost_function, x0, method='BFGS', jac=True, options={'gtol': 1e-6, 'disp': True})
        return self.flat_vector_to_properties(self.initial_float_properties, res.x)

        