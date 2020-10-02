import bpy

from bpy.types import Node
from .._base.node_base import ScNode
from ...helper import focus_on_object

class ScGetParent(Node, ScNode):
    bl_idname = "ScGetParent"
    bl_label = "Get Parent"
    bl_icon = 'CON_CHILDOF'
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Object")
        self.outputs.new("ScNodeSocketObject", "Parent")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.inputs["Object"].default_value == None
        )
    
    def post_execute(self):
        out = super().post_execute()
        out["Parent"] = self.inputs["Object"].default_value.parent
        return out