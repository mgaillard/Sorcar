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
        if curr_tree and hasattr(curr_tree, 'preset_properties'):
            # Display the number of presets available
            nb_presets = len(curr_tree.preset_properties)
            layout.label(text="Preset configurations (" + str(nb_presets) + ")")
            # Display buttons to select one of the configurations
            if nb_presets > 0:
                # Call the preset operator with the index of the configuration to show
                button0 = self.layout.operator('sorcar.select_preset_configuration', text='Preset: Initial')
                button0.preset_index = 0
                # Call the preset operator with the index of the configuration to show
                button1 = self.layout.operator('sorcar.select_preset_configuration', text='Preset 1')
                button1.preset_index = 1
                # Call the preset operator with the index of the configuration to show
                button2 = self.layout.operator('sorcar.select_preset_configuration', text='Preset 2')
                button2.preset_index = 2
