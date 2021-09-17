import bpy

from bpy.types import Panel
from ._base.panel_base import ScPanel

class ScInverseModelingPanel(Panel, ScPanel):
    bl_label = "Inverse Modeling"
    bl_idname = "NODE_PT_sc_inverse_modeling"
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        layout.operator("sorcar.clear_objects")
        layout.operator("sorcar.print_statistics")
        layout.operator("sorcar.inverse_modeling")

        curr_tree = context.space_data.edit_tree
        if curr_tree:
            # Display the number of presets available
            nb_presets = len(curr_tree.preset_properties)
            layout.label(text="Preset configurations (" + str(nb_presets) + ")")
