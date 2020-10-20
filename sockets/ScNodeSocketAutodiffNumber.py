import bpy

from bpy.props import StringProperty
from bpy.types import NodeSocket
from ._base.socket_base import ScNodeSocket
from ..nodes._base.node_base import ScNode

class ScNodeSocketAutodiffNumber(NodeSocket, ScNodeSocket):
    bl_idname = "ScNodeSocketAutodiffNumber"
    bl_label = "AutodiffNumber"
    color = (0.0, 1.0, 1.0, 1.0)

    default_value: StringProperty(default="")
    default_value_update: StringProperty(default="", update=ScNode.update_value)
    default_type: StringProperty(default="AUTODIFF")

    def __init__(self):
        super().__init__()

    def get_label(self):
        if (self.default_value):
            # TODO: show the value instead of the name
            return repr(self.default_value)
        else:
            return "-"