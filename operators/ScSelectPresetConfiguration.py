import bpy

from bpy.types import Operator
from bpy.props import IntProperty
from ..helper import sc_poll_op
from ..debug import log

class ScSelectPresetConfiguration(Operator):
    """Select one of the preset configuration in the current node tree"""
    bl_idname = "sorcar.select_preset_configuration"
    bl_label = "Select preset"

    preset_index: IntProperty(default=0, min=0)

    @classmethod
    def poll(cls, context):
        return sc_poll_op(context)

    def execute(self, context):
        curr_tree = context.space_data.edit_tree
        if curr_tree and hasattr(curr_tree, 'preset_properties'):
            # Check that the provided index refers to a preset that exists
            nb_presets = len(curr_tree.preset_properties)

            if self.preset_index >= 0 and self.preset_index < nb_presets:
                preset = curr_tree.preset_properties[self.preset_index]
                preset_parameters = preset['params']
                preset_label = preset['label']
                # Display the configuration
                print('Selecting the configuration: {}'.format(preset_label))
                # Set the configuration
                curr_tree.set_float_properties(preset_parameters)
                # Reset state
                context.view_layer.update()
            return {'FINISHED'}
        else:
            log("OPERATOR", None, self.bl_idname, "No current edit tree, operation cancelled", 1)
        return {'CANCELLED'}