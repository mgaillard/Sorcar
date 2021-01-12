import bpy

from bpy.props import BoolProperty, PointerProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import remove_object

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
        bpy.ops.object.duplicate(
            linked = self.inputs["Linked"].default_value
        )
    
    def post_execute(self):
        out = super().post_execute()
        
        # Duplicate the object
        self.out_mesh = bpy.context.active_object
        out["Duplicate Object"] = self.out_mesh
        self.id_data.register_object(self.out_mesh)

        # Get the name of the new object
        original_object = self.inputs["Object"].default_value
        original_name = original_object.name
        object_name = self.out_mesh.name

        # Duplicate the bounding box
        self.prop_nodetree.autodiff_variables.duplicate_box(original_name, object_name)
        # Duplicate the axis system
        self.prop_nodetree.autodiff_variables.duplicate_axis_system(original_name, object_name)
        
        # Assign the bounding box to the duplicated object
        self.out_mesh["OBB"] = object_name

        return out
    
    def free(self):
        super().free()
        self.id_data.unregister_object(self.out_mesh)