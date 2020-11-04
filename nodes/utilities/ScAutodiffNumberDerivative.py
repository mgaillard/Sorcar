import bpy

from bpy.props import FloatProperty, PointerProperty
from bpy.types import Node
from .._base.node_base import ScNode

class ScAutodiffNumberDerivative(Node, ScNode):
    bl_idname = "ScAutodiffNumberDerivative"
    bl_label = "Compute Autodiff Derivative"
    bl_icon = 'LINENUMBERS_ON'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketAutodiffNumber", "AutodiffNumber")
        self.inputs.new("ScNodeSocketAutodiffNumber", "Variable")
        self.outputs.new("ScNodeSocketNumber", "Derivative")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["AutodiffNumber"].default_value == ""
            or self.inputs["Variable"].default_value == ""
        )

    def post_execute(self):
        out = super().post_execute()

        value_name = self.inputs["AutodiffNumber"].default_value
        derivation_variable_name = self.inputs["Variable"].default_value

        value_symbol = self.prop_nodetree.autodiff_variables.get_variable_symbol(value_name)

        out["Derivative"] = self.prop_nodetree.autodiff_variables.evaluate_derivative(value_symbol, derivation_variable_name)
        return out
