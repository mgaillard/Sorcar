import bpy

from bpy.props import FloatProperty, PointerProperty
from bpy.types import Node
from ...types.ScAutodiffProperty import ScAutodiffProperty
from .._base.node_base import ScNode

class ScConvertAutodiffNumber(Node, ScNode):
    bl_idname = "ScConvertAutodiffNumber"
    bl_label = "Convert Autodiff Number"
    bl_icon = 'LINENUMBERS_ON'

    prop_float: FloatProperty(update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Autodiff")
        self.outputs.new("ScNodeSocketNumber", "Value")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
    
    def error_condition(self):
        return super().error_condition()
    
    def pre_execute(self):
        super().pre_execute()
        self.prop_float = self.inputs["Autodiff"].default_value.prop_float
    
    def functionality(self):
        super().functionality()

    def post_execute(self):
        out = super().post_execute()
        out["Value"] = self.prop_float
        return out
