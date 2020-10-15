import bpy

from bpy.props import PointerProperty, FloatProperty, StringProperty
from bpy.types import NodeSocket
from ..types.ScAutodiffProperty import ScAutodiffProperty
from ._base.socket_base import ScNodeSocket
from ..nodes._base.node_base import ScNode

class ScNodeSocketAutodiffNumber(NodeSocket, ScNodeSocket):
    bl_idname = "ScNodeSocketAutodiffNumber"
    bl_label = "AutodiffNumber"
    color = (0.0, 1.0, 1.0, 1.0)

    default_value: PointerProperty(type=ScAutodiffProperty)
    default_value_update: PointerProperty(type=ScAutodiffProperty, update=ScNode.update_value)
    default_type: StringProperty(default="AUTODIFF")

    def get_label(self):
        if (self.default_value):
            return repr(self.default_value)
        else:
            return "-"