import bpy

from bpy.types import Operator
from ..helper import sc_poll_op
from ..debug import log

class ScClearObjects(Operator):
    """Clear the objects from the current tree"""
    bl_idname = "sorcar.clear_objects"
    bl_label = "Clear Objects"

    @classmethod
    def poll(cls, context):
        return sc_poll_op(context)

    def execute(self, context):
        curr_tree = context.space_data.edit_tree
        if (curr_tree):
            # Cleanup objects that are generated by the procedural tree
            log("OPERATOR", curr_tree.name, self.bl_idname, "Node=\""+str(curr_tree.node)+"\"", 1)
            curr_tree.node = None
            curr_tree.reset_nodes(True)
            curr_tree.unregister_all_objects()
            # Cleanup blocks that are unused
            for block in bpy.data.meshes:
                if block.users == 0:
                    log("OPERATOR", curr_tree.name, self.bl_idname, "Remove unused data block " + str(block.name), 1)
                    bpy.data.meshes.remove(block)

            return {'FINISHED'}
        else:
            log("OPERATOR", curr_tree.name, self.bl_idname, "No preview node set, operation cancelled", 1)
        return {'CANCELLED'}