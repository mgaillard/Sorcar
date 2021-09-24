import bpy

import math
from bpy.props import StringProperty, PointerProperty, EnumProperty
from bpy.types import Node
from .._base.node_base import ScNode
from ...optimization.ScAutodiffVariableCollection import ScAutodiffVariable

op_items = [
    # (identifier, name, description)
    ("ADD", "X + Y", "Addition"),
    ("SUB", "X - Y", "Subtraction"),
    ("MULT", "X * Y", "Multiplication"),
    ("DIV", "X / Y", "Division"),
    None,
    ("POW", "X ^ Y", "Exponent (Power)"),
    ("LOG", "Log(X)", "Logarithm"),
    ("SQRT", "√X", "Square Root"),
    None,
    ("NEGX", "-X", "Negative X"),
    ("NEGY", "-Y", "Negative Y"),
    ("DIV2", "X / 2", "Division by 2"),
]

class ScAutodiffNumberMathOp(Node, ScNode):
    bl_idname = "ScAutodiffNumberMathOp"
    bl_label = "Autodiff Math Operation"
    bl_icon = 'CON_TRANSLIKE'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_op: EnumProperty(name="Operation", items=op_items, default="ADD", update=ScNode.update_value)
    x_default_name: StringProperty(default="", update=ScNode.update_value)
    y_default_name: StringProperty(default="", update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketAutodiffNumber", "X").init("x_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Y").init("y_default_name", True)
        self.inputs.new("ScNodeSocketString", "Operation").init("in_op", True)
        self.outputs.new("ScNodeSocketAutodiffNumber", "Value")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        # TODO: check input according to the type of operation like in ScMathsOp
        op = self.inputs["Operation"].default_value
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or (not op in [i[0] for i in op_items if i])
        )

    def post_execute(self):
        out = super().post_execute()

        # Rename variables for convenience
        op = self.inputs["Operation"].default_value
        autodiff_variables = self.prop_nodetree.autodiff_variables

        # Read input values
        x_name = self.inputs["X"].default_value
        y_name = self.inputs["Y"].default_value

        # Get corresponding symbols
        x_symbol = self.prop_nodetree.autodiff_variables.get_variable_symbol(x_name)
        y_symbol = self.prop_nodetree.autodiff_variables.get_variable_symbol(y_name)

        # If symbols are not available, replace them with a default value
        default_value = 0.0
        if x_symbol is None:
            x_symbol = autodiff_variables.get_temporary_const_variable(default_value)
        if y_symbol is None:
            y_symbol = autodiff_variables.get_temporary_const_variable(default_value)

        x_value = autodiff_variables.get_variable_value(x_name, 0.0)
        y_value = autodiff_variables.get_variable_value(y_name, 0.0)

        # Compute the operation
        if op == 'ADD':
            result_value = x_value + y_value
            result_symbol = x_symbol + y_symbol
        elif op == 'SUB':
            result_value = x_value - y_value
            result_symbol = x_symbol - y_symbol
        elif op == 'MULT':
            result_value = x_value * y_value
            result_symbol = x_symbol * y_symbol
        elif op == 'DIV':
            result_value = x_value / y_value
            result_symbol = x_symbol / y_symbol
        elif op == 'POW':
            result_value = x_value
            result_symbol = ScAutodiffVariable.pow(x_symbol, y_symbol)
        elif op == 'LOG':
            result_value = x_value
            result_symbol = ScAutodiffVariable.log(x_symbol)
        elif op == 'SQRT':
            result_value = math.sqrt(x_value)
            result_symbol = ScAutodiffVariable.sqrt(x_symbol)
        elif op == 'NEGX':
            result_value = -x_value
            result_symbol = -x_symbol
        elif op == 'NEGY':
            result_value = -y_value
            result_symbol = -y_symbol
        elif op == 'DIV2':
            result_value = x_value / 2.0
            result_symbol = x_symbol / 2.0

        # Register the variable in the tree, it cannot be constant because it needs to be a symbol
        autodiff_variables.create_variable(self.name, False, False, 0.0, 0.0, result_value)
        autodiff_variables.set_variable_symbol(self.name, result_symbol, result_value)

        # Output the name of the variable
        out["Value"] = self.name

        return out