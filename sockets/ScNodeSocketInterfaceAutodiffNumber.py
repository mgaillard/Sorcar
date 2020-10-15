import bpy

from bpy.props import PointerProperty
from bpy.types import NodeSocketInterface
from ..types.ScAutodiffProperty import ScAutodiffProperty
from ._base.interface_base import ScNodeSocketInterface

class ScNodeSocketInterfaceAutodiffNumber(NodeSocketInterface, ScNodeSocketInterface):
    bl_idname = "ScNodeSocketInterfaceAutodiffNumber"
    bl_label = "AutodiffNumber"
    bl_socket_idname = "ScNodeSocketAutodiffNumber"

    color = (0.0, 1.0, 1.0, 1.0)
    default_value: PointerProperty(name="Default Value", type=ScAutodiffProperty)