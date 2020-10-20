import bpy

from bpy.props import StringProperty
from bpy.types import NodeSocketInterface
from ._base.interface_base import ScNodeSocketInterface

class ScNodeSocketInterfaceAutodiffNumber(NodeSocketInterface, ScNodeSocketInterface):
    bl_idname = "ScNodeSocketInterfaceAutodiffNumber"
    bl_label = "AutodiffNumber"
    bl_socket_idname = "ScNodeSocketAutodiffNumber"

    color = (0.0, 1.0, 1.0, 1.0)
    default_value: StringProperty(default="")