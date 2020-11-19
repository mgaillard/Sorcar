import bpy

from bpy.props import FloatProperty, PointerProperty
from bpy.types import Node
from .._base.node_base import ScNode

class ScAutodiffNumberMathOp(Node, ScNode):
    bl_idname = "ScAutodiffNumberMathOp"
    bl_label = "Autodiff Math Operation"
    bl_icon = 'CON_TRANSLIKE'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketAutodiffNumber", "X")
        self.inputs.new("ScNodeSocketAutodiffNumber", "Y")
        self.outputs.new("ScNodeSocketAutodiffNumber", "Value")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["X"].default_value == ""
            or self.inputs["Y"].default_value == ""
        )

    def post_execute(self):
        out = super().post_execute()

        x_name = self.inputs["X"].default_value
        y_name = self.inputs["Y"].default_value

        x_value = self.prop_nodetree.autodiff_variables.get_variable_value(x_name, 0.0)
        y_value = self.prop_nodetree.autodiff_variables.get_variable_value(y_name, 0.0)

        x = self.prop_nodetree.autodiff_variables.get_variable_symbol(x_name)
        y = self.prop_nodetree.autodiff_variables.get_variable_symbol(y_name)

        # Compute the operation
        result_value = x_value + y_value
        result_symbol = x + y

        # Register the variable in the tree
        self.prop_nodetree.autodiff_variables.create_variable(self.name, False, result_value)
        self.prop_nodetree.autodiff_variables.set_variable_symbol(self.name, result_symbol, result_value)

        # Output the name of the variable
        out["Value"] = self.name

        return out