import bpy

from bpy.props import FloatProperty, PointerProperty
from bpy.types import Node
from ...types.ScAutodiffProperty import ScAutodiffProperty
from .._base.node_base import ScNode

class ScAutodiffNumber(Node, ScNode):
    bl_idname = "ScAutodiffNumber"
    bl_label = "Autodiff Number"
    bl_icon = 'LINENUMBERS_ON'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    prop_float: FloatProperty(name="Number", update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.outputs.new("ScNodeSocketAutodiffNumber", "Value")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
        layout.prop(self, "prop_float")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
        )

    def pre_execute(self):
        super().pre_execute()
        # TODO: add special function in the NodeTree to get autodiff variables
        if not hasattr(self.prop_nodetree, "autodiff_variables"):
            self.prop_nodetree.autodiff_variables = {}

    def functionality(self):
        super().functionality()
        if (not self.name in self.prop_nodetree.autodiff_variables):
            self.prop_nodetree.autodiff_variables[self.name] = 0.0
    
    def post_execute(self):
        out = super().post_execute()
        # TODO: add special function in the NodeTree to set autodiff variables
        self.prop_nodetree.autodiff_variables[self.name] = self.prop_float
        out["Value"] = self.name
        return out
