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
            or self.inputs["AutodiffNumber"].default_value == ""
        )

    def functionality(self):
        super().functionality()
        var_name = self.inputs["AutodiffNumber"].default_value
        if (not self.prop_nodetree.autodiff_variables.has(var_name)):
            self.prop_nodetree.autodiff_variables.set_value(var_name, 0.0)

    def post_execute(self):
        out = super().post_execute()

        var_name = self.inputs["AutodiffNumber"].default_value
        variable = self.prop_nodetree.autodiff_variables.get_variable(var_name)
        out["Value"] = float(self.prop_nodetree.autodiff_variables.evaluate_value(variable))
        
        # Return the value directly
        # out["Value"] = float(self.prop_nodetree.autodiff_variables.get_value(var_name, 0.0))

        return out