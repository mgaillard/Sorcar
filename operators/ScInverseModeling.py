import bpy

from bpy.types import Operator
from ..helper import sc_poll_op

class ScInverseModeling(Operator):
    """Ask the user for a modification and propagate it back"""
    bl_idname = "sorcar.inverse_modeling"
    bl_label = "Inverse Modeling"

    def __init__(self):
        print("Start: ScInverseModeling")

    def __del__(self):
        print("End: ScInverseModeling")

    @classmethod
    def poll(cls, context):
        """ Test if the operator can be called or not """
        return sc_poll_op(context)

    def modal(self, context, event):
        if event.type == 'ESC':    
            # TODO: list all objects and their positions after modification
            # Measure the dimensions of the object in the scene
            object_dimensions = context.scene.objects["Object"].dimensions
            cube_new_size = min(object_dimensions.x, object_dimensions.y, object_dimensions.z)
            # Update the scale of the Cube node with the measured scale
            curr_tree = context.space_data.edit_tree
            curr_tree.set_value(node_name="Create Cube", attr_name="in_size", value=cube_new_size)
            # Finished, The operator exited after completing its action.
            return {'FINISHED'}
        # Pass Through: do nothing and pass the event on.
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        """ Execute the operator when invoked by the user """
        # TODO: remove everything in the viewport (reset)
        # We start by executing the current graph
        curr_tree = context.space_data.edit_tree
        if (curr_tree.nodes.active):
            curr_tree.node = curr_tree.nodes.active.name
            curr_tree.execute_node()
            # TODO: list all objects and their positions before modification
            # The graph is executed and the object is ready to be transformed
            print("Instruction: Move the object in the 3D viewport")
            context.window_manager.modal_handler_add(self)
            # We watch modifications made by the user in the modal part of the operator
            # Running Modal: keep the operator running with blender.
            return {"RUNNING_MODAL"}
        
        # The operator exited without doing anything, so no undo entry should be pushed
        return {"CANCELLED"}
