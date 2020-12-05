import bpy

from bpy.props import BoolProperty, FloatProperty, PointerProperty
from bpy.types import Node
from .._base.node_base import ScNode

class ScAutodiffNumber(Node, ScNode):
    bl_idname = "ScAutodiffNumber"
    bl_label = "Autodiff Number"
    bl_icon = 'LINENUMBERS_ON'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    # Whether the variable is constant or not
    prop_const: BoolProperty(name="Constant", update=ScNode.update_value)
    # Whether the result is bounded or not
    prop_bounded: BoolProperty(name="Bounded", update=ScNode.update_value)
    # Minimum value for this variable
    prop_float_min: FloatProperty(name="Min", default=-100.0 ,update=ScNode.update_value)
    # Maximum value for this variable
    prop_float_max: FloatProperty(name="Max", default=100.0, update=ScNode.update_value)
    # Value of the variable
    prop_float: FloatProperty(name="Number", update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.outputs.new("ScNodeSocketAutodiffNumber", "Value")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
        layout.prop(self, "prop_const")
        if not self.prop_const:
            layout.prop(self, "prop_bounded")
            if self.prop_bounded:
                layout.prop(self, "prop_float_min")
                layout.prop(self, "prop_float_max")
        layout.prop(self, "prop_float")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
        )

    def functionality(self):
        super().functionality()
        # If the value is bounded, clamp the value according to the minimum and maximum
        if not self.prop_const and self.prop_bounded:
            if self.prop_float < self.prop_float_min:
                # Warning: assign a new value to prop_float, which re-execute the node
                self.prop_float = self.prop_float_min
            if self.prop_float > self.prop_float_max:
                # Warning: assign a new value to prop_float, which re-execute the node
                self.prop_float = self.prop_float_max
    
    def post_execute(self):
        out = super().post_execute()
        self.prop_nodetree.autodiff_variables.create_variable(self.name, self.prop_const, self.prop_float)
        out["Value"] = self.name
        return out
