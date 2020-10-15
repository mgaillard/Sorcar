import bpy

from bpy.props import FloatProperty, PointerProperty
from bpy.types import Node
from ...types.ScAutodiffProperty import ScAutodiffProperty
from .._base.node_base import ScNode

class ScAutodiffNumber(Node, ScNode):
    bl_idname = "ScAutodiffNumber"
    bl_label = "Autodiff Number"
    bl_icon = 'LINENUMBERS_ON'

    prop_autodiff_float: PointerProperty(name="AutodiffNumber", type=ScAutodiffProperty, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.outputs.new("ScNodeSocketAutodiffNumber", "Value")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self.prop_autodiff_float, "prop_float")
    
    def error_condition(self):
        return (super().error_condition())
    
    def post_execute(self):
        out = super().post_execute()
        out["Value"] = self.prop_autodiff_float
        return out
