import bpy

from bpy.types import Operator
from ..helper import sc_poll_op
from ..optimization.ScInverseModelingSolver import ScInverseModelingSolver
from ..debug import log

class ScInverseModeling(Operator):
    """Ask the user for a modification and propagate it back"""
    bl_idname = "sorcar.inverse_modeling"
    bl_label = "Inverse Modeling"

    @classmethod
    def poll(cls, context):
        """ Test if the operator can be called or not """
        return sc_poll_op(context)

    def __init__(self):
        self.original_bounding_boxes = {}

    def find_target_bounding_boxes(self, new_bounding_boxes):
        """ Find objects that changed and output the target bounding boxes for optimization """
        target_bounding_boxes = {}

        # For each bounding box, compare to the previous box, if it's different then it's part of the target
        for object_name in new_bounding_boxes:
            # Find the corresponding box in the original set of bounding boxes
            if object_name in self.original_bounding_boxes:
                original_box = self.original_bounding_boxes[object_name]
                new_box = new_bounding_boxes[object_name]
                if not(original_box.is_equal(new_box)):
                    # If the bounding box changed, we add it to the set of target bounding boxes
                    target_bounding_boxes[object_name] = new_box
                
        return target_bounding_boxes


    def modal(self, context, event):
        """ Once the user invoked the operator, this function is run until escape is pressed """

        if event.type == 'ESC':
            self.report({'INFO'}, "Inverse modeling complete")
            return {'FINISHED'}


        if event.type == 'MOUSEMOVE':
            curr_tree = context.space_data.edit_tree
            if (curr_tree):
                target_bounding_boxes = self.find_target_bounding_boxes(curr_tree.get_object_boxes())

                if(len(target_bounding_boxes) > 0):
                    
                    #Save selection
                    sel_object_names = [obj.name for obj in bpy.data.objects if obj.select_get()]            

                    # Optimization
                    initial_float_properties = curr_tree.get_float_properties()
                    float_properties_bounds = curr_tree.get_float_properties_bounds()
                    solver = ScInverseModelingSolver(curr_tree,
                                                     target_bounding_boxes,
                                                     initial_float_properties,
                                                     float_properties_bounds,
                                                     context)
                    list_best_parameters = solver.solve()
                    # list_best_parameters is a list of tuples (parameters, label)
                    # By default we choose the first configuration, which is the most optimal one
                    curr_tree.set_float_properties(list_best_parameters[0]['params'])

                    # Reset the list of preset configurations
                    curr_tree.preset_properties = []
                    # Add the initial configuration
                    curr_tree.preset_properties.append({'params': initial_float_properties, 'label': 'Initial'})
                    # Add the all optimal configurations
                    curr_tree.preset_properties = curr_tree.preset_properties + list_best_parameters
                    
                    # Reset state
                    context.view_layer.update()
                    self.original_bounding_boxes = curr_tree.get_object_boxes()

                    #Deselect all
                    for obj in bpy.data.objects:
                        obj.select_set(False)
                    # Renew selection
                    for name in sel_object_names:
                        if name in bpy.data.objects:
                            bpy.data.objects[name].select_set(True)

                return {'PASS_THROUGH'}

        # Pass Through: do nothing and pass the event on.
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        """ Execute the operator when invoked by the user """
        # We start by executing the current graph
        curr_tree = context.space_data.edit_tree
        node = curr_tree.nodes.active
        if (node):
            log("OPERATOR", curr_tree.name, self.bl_idname, "Node=\""+str(node.name)+"\"", 1)
            curr_tree.node = node.name
            curr_tree.execute_node()
            # Update the view
            context.view_layer.update()
            # List all objects and their positions before modification
            self.original_bounding_boxes = curr_tree.get_object_boxes()
            # The graph is executed and the object is ready to be transformed
            context.window_manager.modal_handler_add(self)
            # Tell the user the add-on is in inverse modeling mode
            self.report({'INFO'}, "Inverse modeling mode activated")
            # We watch modifications made by the user in the modal part of the operator
            # Running Modal: keep the operator running with blender.
            return {"RUNNING_MODAL"}
        
        # The operator exited without doing anything, so no undo entry should be pushed
        return {"CANCELLED"}
