import bpy

from bpy.props import PointerProperty, StringProperty
from bpy.types import Node
from .._base.node_base import ScNode
from ...debug import log

class ScSinkNode(Node, ScNode):
    bl_idname = "ScSinkNode"
    bl_label = "Sink Node"
    bl_icon = 'NODETREE'
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketUniversal", "In1")
        self.inputs.new("ScNodeSocketUniversal", "In2")
        self.outputs.new("ScNodeSocketUniversal", "Out1")
        self.outputs.new("ScNodeSocketUniversal", "Out2")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
    
    def post_execute(self):
        out = super().post_execute()
        out["Out1"] = self.inputs["In1"].default_value
        out["Out2"] = self.inputs["In2"].default_value
        return out