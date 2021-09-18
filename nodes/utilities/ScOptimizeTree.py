import bpy

import mathutils
from bpy.props import PointerProperty, StringProperty, FloatVectorProperty
from bpy.types import Node
from .._base.node_base import ScNode
from ...optimization.ScInverseModelingSolver import ScInverseModelingSolver
from ...optimization.ScOrientedBoundingBox import ScOrientedBoundingBox
from ...debug import log
from ...helper import convert_data

class ScOptimizeTree(Node, ScNode):
    bl_idname = "ScOptimizeTree"
    bl_label = "Optimize Tree"
    bl_icon = 'CON_TRANSFORM_CACHE'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketUniversal", "In")
        self.inputs.new("ScNodeSocketArray", "Targets")
        self.inputs.new("ScNodeSocketArray", "Centers")
        self.inputs.new("ScNodeSocketArray", "Axis 0")
        self.inputs.new("ScNodeSocketArray", "Axis 1")
        self.inputs.new("ScNodeSocketArray", "Axis 2")
        self.outputs.new("ScNodeSocketUniversal", "Out")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None            
        )
    
    def functionality(self):
        # Rename variables for convenience
        curr_tree = self.prop_nodetree
        target_names = eval(self.inputs["Targets"].default_value)
        centers = eval(self.inputs["Centers"].default_value)
        axes_0 = eval(self.inputs["Axis 0"].default_value)
        axes_1 = eval(self.inputs["Axis 1"].default_value)
        axes_2 = eval(self.inputs["Axis 2"].default_value)

        log(self.id_data.name, self.name, "functionality", "Optimize tree: " + curr_tree.name)

        # If target_names is just a string, make it an array
        if not isinstance(target_names, list):
            target_names = [target_names]

        # If centers is just a vector, make it an array
        if not isinstance(centers, list):
            centers = [centers]

        # If axes_1 is just a vector, make it an array
        if not isinstance(axes_0, list):
            axis_0 = [axes_0]

        # If axes_2 is just a vector, make it an array
        if not isinstance(axes_1, list):
            axes_1 = [axes_1]
        
        # If axies_3 is just a vector, make it an array
        if not isinstance(axes_2, list):
            axes_2 = [axes_2]

        # Setup target for optimization
        target_bounding_boxes = {}
        tree_boxes = curr_tree.get_object_boxes()
        number_boxes = min(len(target_names), len(centers), len(axes_0), len(axes_1), len(axes_2))

        for i in range(number_boxes):
            target_name = target_names[i]
            # Convert the data to a Vector
            (center_converted, center_tuple) = convert_data(centers[i], from_type="STRING", to_type="VECTOR")
            (axis_0_converted, axis_0_tuple) = convert_data(axes_0[i], from_type="STRING", to_type="VECTOR")
            (axis_1_converted, axis_1_tuple) = convert_data(axes_1[i], from_type="STRING", to_type="VECTOR")
            (axis_2_converted, axis_2_tuple) = convert_data(axes_2[i], from_type="STRING", to_type="VECTOR")
            if center_converted and axis_0_converted and axis_1_converted and axis_2_converted:
                center = mathutils.Vector(center_tuple)
                axis_0 = mathutils.Vector(axis_0_tuple).normalized()
                axis_1 = mathutils.Vector(axis_1_tuple).normalized()
                axis_2 = mathutils.Vector(axis_2_tuple).normalized()
                # Get the current bounding box from the tree
                if target_name in tree_boxes:
                    target_bounding_boxes[target_name] = tree_boxes[target_name]
                else:
                    target_bounding_boxes[target_name] = ScOrientedBoundingBox.defaultBoundingBox()
                # Modify the bounding box accroding to the input to this node
                target_bounding_boxes[target_name].center = center
                target_bounding_boxes[target_name].axis[0] = axis_0
                target_bounding_boxes[target_name].axis[1] = axis_1
                target_bounding_boxes[target_name].axis[2] = axis_2

        # Optimization
        initial_float_properties = curr_tree.get_float_properties()
        float_properties_bounds = curr_tree.get_float_properties_bounds()
        solver = ScInverseModelingSolver(curr_tree, target_bounding_boxes, initial_float_properties, float_properties_bounds)
        list_best_parameters = solver.solve()
        curr_tree.set_float_properties(list_best_parameters[0]['params'])
    
    def post_execute(self):
        out = super().post_execute()
        out["Out"] = self.inputs["In"].default_value
        return out