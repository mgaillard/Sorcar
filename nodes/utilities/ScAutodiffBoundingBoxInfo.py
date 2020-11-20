import bpy

from bpy.types import Node
from bpy.props import PointerProperty
from .._base.node_base import ScNode
from ...helper import focus_on_object
from ...debug import log

class ScAutodiffBoundingBoxInfo(Node, ScNode):
    bl_idname = "ScAutodiffBoundingBoxInfo"
    bl_label = "Autodiff Bounding Box Info"
    bl_icon = 'FILE_3D'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Object")

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["Object"].default_value == None
        )

    def pre_execute(self):
        super().pre_execute()
        focus_on_object(self.inputs["Object"].default_value, True)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def post_execute(self):
        out = super().post_execute()

        # Rename variables for convenience
        current_object = self.inputs["Object"].default_value
        objects = self.prop_nodetree.objects
        autodiff_variables = self.prop_nodetree.autodiff_variables

        # Get the bounding box if it exists
        if "OBB" in current_object:
            box_name = current_object["OBB"]
            transformed_box = autodiff_variables.compute_transformed_bounding_box(objects, box_name)
            box_points = transformed_box.list_points_to_match()
            for box_point in box_points:
                value = autodiff_variables.evaluate_vector(box_point)
                log("ScAutodiffBoundingBoxInfo", None, "box_point", repr(value), level=1)
        
        return out