import bpy

from bpy.props import FloatProperty, PointerProperty
from bpy.types import Node
from .._base.node_base import ScNode

class ScAutodiffNumberConvert(Node, ScNode):
    bl_idname = "ScAutodiffNumberConvert"
    bl_label = "Convert Autodiff Number"
    bl_icon = 'LINENUMBERS_ON'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketAutodiffNumber", "AutodiffNumber")
        self.outputs.new("ScNodeSocketNumber", "Value")
    
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

        var_name = self.inputs["AutodiffNumber"].default_value

        value = 0.0
        if self.prop_nodetree.autodiff_variables.has_variable(var_name):
            variable_symbol = self.prop_nodetree.autodiff_variables.get_variable_symbol(var_name)
            value = self.prop_nodetree.autodiff_variables.evaluate_value(variable_symbol)
        
        out["Value"] = value

        return out
