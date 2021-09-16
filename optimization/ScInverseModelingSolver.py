import bpy

from time import perf_counter
import numpy as np
from scipy.optimize import minimize, Bounds
from ..debug import log
from ..experiments.Functions import CasadiFunction

class ScInverseModelingSolver:

    def __init__(self, curr_tree, target_bounding_boxes, initial_float_properties, float_properties_bounds, context=None):
        self.curr_tree = curr_tree
        self.target_bounding_boxes = target_bounding_boxes
        self.initial_float_properties = initial_float_properties
        self.float_properties_bounds = float_properties_bounds
        self.property_map = self.create_properties_map(self.initial_float_properties)
        self.context = context


    @staticmethod
    def create_properties_map(float_properties):
        property_map = []
        for property_name in float_properties:
            property_map.append(property_name)
        return property_map


    @staticmethod
    def create_properties_map_from_symbols(symbols):
        property_map = []
        for symbol in symbols:
            property_map.append(symbol)
        return property_map


    @staticmethod
    def is_problem_unconstrained(float_properties_bounds):
        for property_name in float_properties_bounds:
            if float_properties_bounds[property_name]["bounded"]:
                return False
        return True

    @staticmethod
    def properties_bounds_to_flat_vector(property_map, float_properties_bounds):
        lb = []
        ub = []
        for i in range(len(property_map)):
            property_name = property_map[i]
            if property_name in float_properties_bounds:
                if float_properties_bounds[property_name]["bounded"]:
                    lb.append(float_properties_bounds[property_name]["min"])
                    ub.append(float_properties_bounds[property_name]["max"])
                else:
                    lb.append(-np.inf)
                    ub.append(np.inf)
            else:
                lb.append(-np.inf)
                ub.append(np.inf)
        return Bounds(np.array(lb), np.array(ub))
    

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


    def compute_error(self, bounding_boxes):
        error = 0.0

        # For each bounding_boxes compare the to the target
        for object_name in bounding_boxes:
            # Find the corresponding box in the target
            if object_name in self.target_bounding_boxes:
                # List points to match between the two objects
                target_box_points = self.target_bounding_boxes[object_name].list_points_to_match()
                box_points = bounding_boxes[object_name].list_points_to_match()
                # The error is the sum of square distances between corners of the bounding boxes
                number_points = min(len(target_box_points), len(box_points))
                for i in range(number_points):
                    error = error + (target_box_points[i].x - box_points[i].x)**2
                    error = error + (target_box_points[i].y - box_points[i].y)**2
                    error = error + (target_box_points[i].z - box_points[i].z)**2
        
        return error

    
    def evaluate_cost_function(self, x):
        """
        Evaluate the cost function
        Updates the parameters in the graph and executes it
        Potentially slow
        """
        # Update parameters in the tree
        float_properties = self.flat_vector_to_properties(self.property_map, x)
        self.curr_tree.set_float_properties(float_properties)
        self.curr_tree.execute_node()
        # Update the view
        self.context.view_layer.update()
        # Collect the name of bounding boxes
        bounding_boxes = self.curr_tree.get_object_boxes()
        # Compute the error
        error = self.compute_error(bounding_boxes)
        # Lof the result without gradient
        log("ScInverseModelingSolver", None, "evaluate_cost_function", "f:{}".format(repr(error)), level=1)
        return error


    def solve_with_autodiff(self):
        """
        Solve the problem using autodiff to compute the gradient
        """
        # Just a reference for more concise code
        autodiff_variables = self.curr_tree.autodiff_variables
        # Execute the graph with the initial parameters
        self.curr_tree.set_float_properties(self.initial_float_properties)
        # Start measuring optimization time
        time_start = perf_counter()
        # Collect the name of autodiff bounding boxes
        bounding_boxes = self.curr_tree.get_object_autodiff_boxes_names()
        # Build the cost function and save it for later
        autodiff_cost_function = autodiff_variables.build_cost_function(self.target_bounding_boxes, bounding_boxes, self.curr_tree.objects)
        # Build functions with concatenated vectors to optimize with 1D vectors of parameters
        func = autodiff_variables.build_function(autodiff_cost_function, vertcat_symbols=True)
        grad_func = autodiff_variables.build_gradient(autodiff_cost_function, vertcat_symbols=True)
        # Wrap the function and its gradient in a CasadiFunction object
        cost_function = CasadiFunction()
        cost_function.set_functions(func, grad_func)
        # Adapt the property map to have the same order as symbols in the objective function
        func_property_map = self.create_properties_map_from_symbols(autodiff_variables.get_symbols(autodiff_cost_function))
        # Build the initial parameter vector x0
        x0 = self.properties_to_flat_vector(func_property_map, self.initial_float_properties)
        if self.is_problem_unconstrained(self.float_properties_bounds):
            # Use the "BFGS" solver on unconstrained problem
            res = minimize(cost_function.evaluate, x0, method='BFGS', jac=cost_function.derivative, options={'gtol': 1e-6, 'disp': True})
        else:
            # Use the "L-BFGS-B" solver and define bounds
            bounds = self.properties_bounds_to_flat_vector(func_property_map, self.float_properties_bounds)
            res = minimize(cost_function.evaluate, x0, method='L-BFGS-B', jac=cost_function.derivative, bounds=bounds, options={'gtol': 1e-6, 'disp': True})
        time_end = perf_counter()
        log("ScInverseModelingSolver", None, "solve", "Execution time: " + str(time_end - time_start), level=1)
        return self.flat_vector_to_properties(func_property_map, res.x)


    def solve_without_autodiff(self):
        """
        Solve the problem without using autodiff
        Finite differences are used instead to estimate the gradient
        Works well when the number of parameters is not too high
        """
        # Start measuring optimization time
        time_start = perf_counter()
        x0 = self.properties_to_flat_vector(self.property_map, self.initial_float_properties)
        if self.is_problem_unconstrained(self.float_properties_bounds):
            # Use the "Nelder-Mead" solver on unconstrained problem
            res = minimize(self.evaluate_cost_function, x0, method='nelder-mead', options={'xatol': 1e-1, 'disp': True})
        else:
            # Use the "Powell" solver and define bounds
            bounds = self.properties_bounds_to_flat_vector(self.property_map, self.float_properties_bounds)
            res = minimize(self.evaluate_cost_function, x0, method='Powell', bounds=bounds, options={'xtol': 1e-1, 'disp': True})
        
        time_end = perf_counter()
        log("ScInverseModelingSolver", None, "solve", "Execution time: " + str(time_end - time_start), level=1)
        return self.flat_vector_to_properties(self.property_map, res.x)
    

    def solve(self):
        """
        Solve the optimization problem
        Select between the autodiff or finite differences optimizer 
        """
        # Check if the procedural tree is autodifferentiable
        # Only check the target bounding boxes, if some of the boxes are not autodifferentiable
        # but they are not part of the target, we can still benefit from autodiff acceleration
        if self.curr_tree.are_target_boxes_all_autodiff(self.target_bounding_boxes):
            return self.solve_with_autodiff()
        else:
            return self.solve_without_autodiff()
        
        
