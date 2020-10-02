import bpy

from bpy.types import Operator
from ..helper import sc_poll_op
from ..debug import log

class ScInverseModeling(Operator):
    """Ask the user for a modification and propagate it back"""
    bl_idname = "sorcar.inverse_modeling"
    bl_label = "Inverse Modeling"

    @classmethod
    def poll(cls, context):
        """ Test if the operator can be called or not """
        return sc_poll_op(context)

    def modal(self, context, event):
        if event.type == 'ESC':
            # List all objects and their positions before modification
            curr_tree = context.space_data.edit_tree
            if (curr_tree):
                bouding_boxes = curr_tree.get_object_boxes()
            '''
            # Measure the dimensions of the object in the scene
            object_dimensions = context.scene.objects["Object"].dimensions
            cube_new_size = min(object_dimensions.x, object_dimensions.y, object_dimensions.z)
            # Update the scale of the Cube node with the measured scale
            curr_tree = context.space_data.edit_tree
            curr_tree.set_value(node_name="Create Cube", attr_name="in_size", value=cube_new_size)
            '''
            # Finished, The operator exited after completing its action.
            return {'FINISHED'}
        # Pass Through: do nothing and pass the event on.
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        """ Execute the operator when invoked by the user """
        # We start by executing the current grap h
        curr_tree = context.space_data.edit_tree
        node = curr_tree.nodes.active
        if (node):
            log("OPERATOR", curr_tree.name, self.bl_idname, "Node=\""+str(node.name)+"\"", 1)
            curr_tree.node = node.name
            curr_tree.execute_node()
            # List all parameters in the tree (all float number nodes)
            tree_properties = curr_tree.get_float_properties()
            # List all objects and their positions before modification
            bouding_boxes = curr_tree.get_object_boxes()
            # The graph is executed and the object is ready to be transformed
            context.window_manager.modal_handler_add(self)
            # We watch modifications made by the user in the modal part of the operator
            # Running Modal: keep the operator running with blender.
            return {"RUNNING_MODAL"}
        
        # The operator exited without doing anything, so no undo entry should be pushed
        return {"CANCELLED"}
