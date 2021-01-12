import bpy

from bpy.props import PointerProperty, BoolProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import focus_on_object, remove_object, sc_poll_mesh, apply_all_modifiers
from ...optimization.ScOrientedBoundingBox import ScOrientedBoundingBox

class ScAutodiffConvertObject(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffConvertObject"
    bl_label = "Autodiff Convert Object"
    bl_icon = 'EYEDROPPER'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    
    def init(self, context):
        super().init(context)

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
        )
    
    def post_execute(self):
        out = super().post_execute()

        out_mesh = bpy.context.active_object
        object_name = out_mesh.name

        # Apply modifications so that the base matrix is the identity
        apply_all_modifiers(out_mesh)
        bpy.ops.object.transform_apply(location = True, rotation = True, scale = True)
        # Measure dimensions of the Object
        object_bounding_box = ScOrientedBoundingBox.fromObject(out_mesh)

        # Register a constant autodiff bounding box for the object        
        self.prop_nodetree.autodiff_variables.create_default_axis_system(object_name)
        self.prop_nodetree.autodiff_variables.set_box_from_constants(object_name, object_bounding_box)
        out_mesh["OBB"] = object_name

        out["Object"] = out_mesh

        return out