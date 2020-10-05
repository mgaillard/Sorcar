import bpy

import numpy as np
from scipy.optimize import minimize
from ..debug import log

class ScInverseModelingSolver:

    def __init__(self, curr_tree, target_bounding_boxes, initial_float_properties):
        self.curr_tree = curr_tree
        self.target_bounding_boxes = target_bounding_boxes
        self.initial_float_properties = initial_float_properties


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


    def compute_error(self, bounding_boxes):
        error = 0.0

        # For each bounding_boxes compare the to the target
        for object_name in bounding_boxes:
            # Find the corresponding box in the target
            if object_name in self.target_bounding_boxes:
                target_box = self.target_bounding_boxes[object_name]
                box = bounding_boxes[object_name]
                # The error is the sum of square distances between corners of the bounding boxes
                error = error + (target_box["min"].x - box["min"].x)**2
                error = error + (target_box["min"].y - box["min"].y)**2
                error = error + (target_box["min"].z - box["min"].z)**2
                error = error + (target_box["max"].x - box["max"].x)**2
                error = error + (target_box["max"].y - box["max"].y)**2
                error = error + (target_box["max"].z - box["max"].z)**2
        
        return error

    
    def cost_function(self, x):
        # Update parameters in the tree
        float_properties = self.flat_vector_to_properties(self.initial_float_properties, x)
        self.curr_tree.set_float_properties(float_properties)
        self.curr_tree.execute_node()
        # Compute the error
        return self.compute_error(self.curr_tree.get_object_boxes())

    
    def solve(self):
        x0 = self.properties_to_flat_vector(self.initial_float_properties)
        res = minimize(self.cost_function, x0, method='nelder-mead', options={'xatol': 1e-1, 'disp': True})
        print(res.x)