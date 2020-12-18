import bpy

import mathutils
from bpy.props import PointerProperty, StringProperty, FloatVectorProperty
from bpy.types import Node
from .._base.node_base import ScNode
from ...optimization.ScInverseModelingSolver import ScInverseModelingSolver
from ...optimization.ScOrientedBoundingBox import ScOrientedBoundingBox
from ...debug import log

class ScOptimizeTree(Node, ScNode):
    bl_idname = "ScOptimizeTree"
    bl_label = "Optimize Tree"
    bl_icon = 'CON_TRANSFORM_CACHE'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_target_name: StringProperty(name="Target name", update=ScNode.update_value)
    in_center: FloatVectorProperty(name="Center", update=ScNode.update_value)
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketUniversal", "In")
        self.inputs.new("ScNodeSocketString", "Target name").init("in_target_name", True)
        self.inputs.new("ScNodeSocketVector", "Center").init("in_center", True)
        self.outputs.new("ScNodeSocketUniversal", "Out")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["Target name"].default_value == ""
        )
    
    def functionality(self):
        # Rename tree for convenience
        curr_tree = self.prop_nodetree
        target_name = self.inputs["Target name"].default_value
        center = mathutils.Vector(self.inputs["Center"].default_value)

        log(self.id_data.name, self.name, "functionality", "Optimize tree: " + curr_tree.name)

        # Setup target for optimization
        target_bounding_boxes = {}
        # Get the current bounding box from the tree
        tree_boxes = curr_tree.get_object_boxes()
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