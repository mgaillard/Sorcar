import bpy

from bpy.props import PointerProperty, BoolProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_input import ScInputNode

import traceback

class ScCreateAutodiffCube(Node, ScInputNode):
    bl_idname = "ScCreateAutodiffCube"
    bl_label = "Create Autodiff Cube"
    bl_icon = 'MESH_CUBE'
    
    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_uv: BoolProperty(default=True, update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketBool", "Generate UVs").init("in_uv")
        self.inputs.new("ScNodeSocketAutodiffNumber", "Size")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")

    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["Size"].default_value == ""
            # TODO: check that Size value is > 0.0
        )
    
    def functionality(self):
        super().functionality()

        size_name = self.inputs["Size"].default_value
        size_value = self.prop_nodetree.autodiff_variables.get_value(size_name, 1.0)
        size_symbol = self.prop_nodetree.autodiff_variables.get_variable(size_name)

        bpy.ops.mesh.primitive_cube_add(
            size = size_value,
            calc_uvs = self.inputs["Generate UVs"].default_value
        )

        # TODO: if the size input is not valid, output a constant OBB

        # Compute extent of cube
        extent_symbol = [
            size_symbol / 2.0,
            size_symbol / 2.0,
            size_symbol / 2.0
        ]

        # Register the bounding box
        object_name = self.inputs["Name"].default_value
        self.prop_nodetree.autodiff_variables.set_box(object_name, extent_symbol)
        bpy.context.active_object["OBB"] = object_name