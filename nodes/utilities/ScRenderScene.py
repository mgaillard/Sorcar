import bpy

from bpy.props import StringProperty, FloatProperty
from bpy.types import Node
from .._base.node_base import ScNode
from ...debug import log

class ScRenderScene(Node, ScNode):
    bl_idname = "ScRenderScene"
    bl_label = "Render Scene"
    bl_icon = 'OUTLINER_OB_LIGHTPROBE'

    in_filename: StringProperty(default="C:\\tmp\\render.png", update=ScNode.update_value)
    in_image_width: FloatProperty(default=1920, update=ScNode.update_value)
    in_image_height: FloatProperty(default=1080, update=ScNode.update_value)
    
    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketUniversal", "In")
        self.inputs.new("ScNodeSocketString", "Filename").init("in_filename", True)
        self.inputs.new("ScNodeSocketNumber", "Width").init("in_image_width", True)
        self.inputs.new("ScNodeSocketNumber", "Height").init("in_image_height", True)
        self.outputs.new("ScNodeSocketUniversal", "Out")
    
    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)

    def functionality(self):
        # Read input
        filename = self.inputs["Filename"].default_value
        width = int(self.inputs["Width"].default_value)
        height = int(self.inputs["Height"].default_value)

        # Render
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.context.scene.render.filepath = filename
        bpy.ops.render.render(write_still=True)
    
    def post_execute(self):
        out = super().post_execute()
        out["Out"] = self.inputs["In"].default_value
        return out