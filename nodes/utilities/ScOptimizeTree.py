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

        log(self.id_data.name, self.name, "functionality", "Optimize tree: " + curr_tree.name)

        # If target_names is just a string, make it an array
        if not isinstance(target_names, list):
            target_names = [target_names]

        # If centers is just a vector, make it an array
        if not isinstance(centers, list):
            centers = [centers]

        # Setup target for optimization
        target_bounding_boxes = {}
        tree_boxes = curr_tree.get_object_boxes()

        for i in range(min(len(target_names), len(centers))):
            target_name = target_names[i]
            # Convert the center to a Vector
            (converted, center_tuple) = convert_data(centers[i], from_type="STRING", to_type="VECTOR")
            if converted:
                center = mathutils.Vector(center_tuple)
                # Get the current bounding box from the tree
                if target_name in tree_boxes:
                    target_bounding_boxes[target_name] = tree_boxes[target_name]
                else:
                    target_bounding_boxes[target_name] = ScOrientedBoundingBox.defaultBoundingBox()
                # Modify the bounding box accroding to the input to this node
                target_bounding_boxes[target_name].center = center

        # Optimization
        initial_float_properties = curr_tree.get_float_properties()
        float_properties_bounds = curr_tree.get_float_properties_bounds()
        solver = ScInverseModelingSolver(curr_tree, target_bounding_boxes, initial_float_properties, float_properties_bounds)
        best_float_properties = solver.solve()
        curr_tree.set_float_properties(best_float_properties)
    
    def post_execute(self):
        out = super().post_execute()
        out["Out"] = self.inputs["In"].default_value
        return out