import bpy

from bpy.types import PropertyGroup
from bpy.props import FloatProperty
from ..debug import log

class ScAutodiffProperty(PropertyGroup):
    bl_idname = 'ScAutodiffProperty'
    bl_label = "Autodiff Property"

    prop_float: FloatProperty(name="Float")

    def __repr__(self):
        return 'AutodiffNumber({:.2f})'.format(self.prop_float)
