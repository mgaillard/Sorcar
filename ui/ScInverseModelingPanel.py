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
        # Inverse modeling operator with 0 budget (local optimization only)
        button_inverse_local = layout.operator("sorcar.inverse_modeling", text="Inverse Modeling (local only)")
        button_inverse_local.optimizer_budget = 0
        # Inverse modeling operator with 400 budget
        button_inverse_local = layout.operator("sorcar.inverse_modeling", text="Inverse Modeling")
        button_inverse_local.optimizer_budget = 400
        # Inverse modeling operator with 2000 budget
        button_inverse_local = layout.operator("sorcar.inverse_modeling", text="Inverse Modeling (high budget)")
        button_inverse_local.optimizer_budget = 2000

        curr_tree = context.space_data.edit_tree
        if curr_tree and hasattr(curr_tree, 'preset_properties'):
            # Display the number of presets available
            nb_presets = len(curr_tree.preset_properties)
            
            # Display buttons to go to previous/next configuration
            if nb_presets > 0:
                layout.label(text="Navigation in presets")
                button_prev = self.layout.operator('sorcar.select_preset_configuration', text="Previous preset")
                button_prev.preset_index = max(0, curr_tree.current_preset_index - 1)
                button_next = self.layout.operator('sorcar.select_preset_configuration', text="Next preset")
                button_next.preset_index = min(len(curr_tree.preset_properties) - 1, curr_tree.current_preset_index + 1)
            
            layout.label(text="Preset configurations (" + str(nb_presets) + ")")
            # Display buttons to select one of the configurations
            for i in range(len(curr_tree.preset_properties)):
                preset_label = curr_tree.preset_properties[i]['label']
                # Call the preset operator with the index of the configuration to show
                button = self.layout.operator('sorcar.select_preset_configuration', text=preset_label)
                button.preset_index = i
