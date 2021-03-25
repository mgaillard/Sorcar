import bpy

from bpy.props import BoolProperty, PointerProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import remove_object

from ...optimization import ScInstanceUtils as instance_utils

class ScAutodiffDuplicateObject(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffDuplicateObject"
    bl_label = "Autodiff Duplicate Object"
    bl_icon = 'DUPLICATE'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_linked: BoolProperty(update=ScNode.update_value)
    out_mesh: PointerProperty(type=bpy.types.Object)
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketBool", "Linked").init("in_linked")
        self.outputs.new("ScNodeSocketObject", "Duplicate Object")

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")

    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
        )
    
    def functionality(self):
        super().functionality()
        

        current_object = self.inputs["Object"].default_value
        autodiff_variables = self.prop_nodetree.autodiff_variables
        self.instance = instance_utils.create_N_instances(autodiff_variables, current_object, 1)[0]
        instance_utils.register_recursive(self.instance, self.id_data)
    
    def post_execute(self):
        out = super().post_execute()
        out["Duplicate Object"] = self.instance
        return out
    
    def free(self):
        super().free()
        self.id_data.unregister_object(self.out_mesh)