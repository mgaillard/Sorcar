import bpy

from bpy.props import PointerProperty, StringProperty
from bpy.types import Node
from .._base.node_base import ScNode
from ...debug import log

class ScUpdateTree(Node, ScNode):
    bl_idname = "ScUpdateTree"
    bl_label = "Update Tree"
    bl_icon = 'SCENE_DATA'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_node_name: StringProperty(name="Node name", update=ScNode.update_value)
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketUniversal", "In")
        self.inputs.new("ScNodeSocketString", "Node name").init("in_node_name", True)
        self.outputs.new("ScNodeSocketUniversal", "Out")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["Node name"].default_value == ""
        )
    
    def functionality(self):
        # Rename tree for convenience
        curr_tree = self.prop_nodetree
        node_name = self.inputs["Node name"].default_value

        log(self.id_data.name, self.name, "functionality", "Update tree: " + curr_tree.name)
        
        # Update the view
        curr_tree.set_preview(node_name)
    
    def post_execute(self):
        out = super().post_execute()
        out["Out"] = self.inputs["In"].default_value
        return out