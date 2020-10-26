import bpy

from bpy.props import PointerProperty, BoolProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_input import ScInputNode
from ...helper import focus_on_object, remove_object, sc_poll_mesh, apply_all_modifiers
from ...optimization.ScOrientedBoundingBox import ScOrientedBoundingBox

class ScAutodiffCustomObject(Node, ScInputNode):
    bl_idname = "ScAutodiffCustomObject"
    bl_label = "Autodiff Custom Object"
    bl_icon = 'EYEDROPPER'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_obj: PointerProperty(type=bpy.types.Object, poll=sc_poll_mesh, update=ScNode.update_value)
    in_hide: BoolProperty(default=True, update=ScNode.update_value)
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Object").init("in_obj", True)
        self.inputs.new("ScNodeSocketBool", "Hide Original").init("in_hide")

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
        self.inputs["Object"].default_value.hide_set(False)
        focus_on_object(self.inputs["Object"].default_value)
    
    def functionality(self):
        super().functionality()
        bpy.ops.object.duplicate()
    
    def post_execute(self):
        out = super().post_execute()
        apply_all_modifiers(self.out_mesh)
        self.inputs["Object"].default_value.hide_set(self.inputs["Hide Original"].default_value)

        # Measure dimensions of the Object
        object_bounding_box = ScOrientedBoundingBox.fromObject(self.out_mesh)

        # Register a constant autodiff bounding box for the object
        object_name = self.out_mesh.name
        self.prop_nodetree.autodiff_variables.set_box_from_constants(object_name, object_bounding_box)
        self.out_mesh["OBB"] = object_name

        return out
    
    def free(self):
        super().free()
        if (self.inputs["Object"].default_value):
            self.inputs["Object"].default_value.hide_set(False)