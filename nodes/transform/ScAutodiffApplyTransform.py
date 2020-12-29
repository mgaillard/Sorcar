import bpy

from bpy.props import PointerProperty, BoolProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import convert_array_to_matrix

class ScAutodiffApplyTransform(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffApplyTransform"
    bl_label = "Autodiff Apply Transform"

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_location: BoolProperty(default=True, update=ScNode.update_value)
    in_rotation: BoolProperty(default=True, update=ScNode.update_value)
    in_scale: BoolProperty(default=True, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketBool", "Location").init("in_location", True)
        self.inputs.new("ScNodeSocketBool", "Rotation").init("in_rotation", True)
        self.inputs.new("ScNodeSocketBool", "Scale").init("in_scale", True)

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")

    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
        )
    
    def functionality(self):
        super().functionality()
        
        # Rename variables for convenience
        current_object = self.inputs["Object"].default_value
        apply_location = self.inputs["Location"].default_value
        apply_rotation = self.inputs["Rotation"].default_value
        apply_scale = self.inputs["Scale"].default_value
        autodiff_variables = self.prop_nodetree.autodiff_variables

        # TODO: separate transformations, and remove this condition
        if apply_location and apply_rotation and apply_scale:
            # Perform the operation on the object
            bpy.ops.object.transform_apply(
                location = apply_location,
                rotation = apply_rotation,
                scale = apply_scale
            )

        # Get the bounding box if it exists, then transforms it
        if "OBB" in current_object:
            box_name = current_object["OBB"]

            # TODO: separate transformations
            if apply_location and apply_rotation and apply_scale:
                # Get the current bounding box transformed according to its axis system
                # Since we just want the current axis system applied on the bounding box and not 
                # the full hierarchy, we pass an empty list of objects
                transformed_box = autodiff_variables.compute_transformed_bounding_box([], box_name)
                # Modify the box
                autodiff_variables.set_box(box_name, transformed_box)
                # Reset the axis system to default
                self.prop_nodetree.autodiff_variables.create_default_axis_system(box_name)
                # It's not necessary to evaluate the world matrix and update it, since it has been reset to default