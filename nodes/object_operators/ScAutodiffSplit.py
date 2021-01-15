import bpy

from mathutils import Vector
from bpy.props import PointerProperty, StringProperty, EnumProperty, IntProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import focus_on_object, remove_object, convert_array_to_matrix
from ...debug import log

class ScAutodiffSplit(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffSplit"
    bl_label = "Autodiff split"
    bl_icon = 'OUTLINER_OB_POINTCLOUD'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_obj: PointerProperty(type=bpy.types.Object, update=ScNode.update_value)
    in_repeat_0: IntProperty(default=1, min=1, max=100, update=ScNode.update_value)
    in_repeat_1: IntProperty(default=1, min=1, max=100, update=ScNode.update_value)
    prop_obj_array: StringProperty(default="[]")

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Scatter Object").init("in_obj", True)
        self.inputs.new("ScNodeSocketNumber", "Repeat0").init("in_repeat_0", True)
        self.inputs.new("ScNodeSocketNumber", "Repeat1").init("in_repeat_1", True)
        self.outputs.new("ScNodeSocketArray", "Scattered Objects")
    

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
        
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["Scatter Object"].default_value == None
        )
    
    def pre_execute(self):
        super().pre_execute()
        self.prop_obj_array = "[]"
    
    def functionality(self):
        super().functionality()

        # Rename variables for convenience
        repeat_0 = int(self.inputs["Repeat0"].default_value)
        repeat_1 = int(self.inputs["Repeat1"].default_value)
        current_object = self.inputs["Object"].default_value
        scatter_object = self.inputs["Scatter Object"].default_value
        autodiff_variables = self.prop_nodetree.autodiff_variables

        # Move the scatter object
        if ("OBB" in scatter_object) and ("OBB" in current_object):
            # Compute symbol of the top coordinate of the current bounding box
            current_box_name = current_object["OBB"]
            # Transform the bounding box according to its own axis system without taking in account the full hierarchy
            current_box = autodiff_variables.compute_transformed_bounding_box([], current_box_name)
            current_center_x = current_box.get_center_x()
            current_center_y = current_box.get_center_y()
            current_center_z = current_box.get_center_z()
            current_extent_x = current_box.get_extent_x()
            current_extent_y = current_box.get_extent_y()
            current_extent_z = current_box.get_extent_z()

            # Compute symbol of the top coordinate of the parent bounding box
            scatter_box_name = scatter_object["OBB"]
            scatter_box = autodiff_variables.get_box(scatter_box_name)
            scatter_extent_z = scatter_box.get_extent_z()

            # Duplicate repeat_0 * repeat_1 versions of the scatter object
            for i in range(repeat_0):
                for j in range(repeat_1):
                    
                    # Duplicate the scatter object if needed
                    if i > 0 or j > 0:
                        focus_on_object(scatter_object)
                        bpy.ops.object.duplicate()
                        new_scatter_object = bpy.context.active_object
                        # Register the new object
                        self.id_data.register_object(new_scatter_object)
                        # Get the name of the new object
                        original_name = scatter_object.name
                        new_scatter_object_name = new_scatter_object.name
                        # Duplicate the bounding box
                        self.prop_nodetree.autodiff_variables.duplicate_box(original_name, new_scatter_object_name)
                        # Duplicate the axis system
                        self.prop_nodetree.autodiff_variables.duplicate_axis_system(original_name, new_scatter_object_name)
                        # Assign the bounding box to the duplicated object
                        new_scatter_object["OBB"] = new_scatter_object_name
                    else:
                        new_scatter_object = scatter_object
                        new_scatter_object_name = new_scatter_object.name
                    
                    # Add the new object to the output array
                    temp = eval(self.prop_obj_array)
                    temp.append(new_scatter_object)
                    self.prop_obj_array = repr(temp)

                    # Useful constants
                    const_i = autodiff_variables.get_temporary_const_variable(i)
                    const_j = autodiff_variables.get_temporary_const_variable(j)

                    # Move the bounding box of the duplicated object to the right location
                    scatter_axis_sytem = autodiff_variables.get_axis_system(new_scatter_object_name)
                    # Set the scale of the scatter object axis system
                    scatter_scale_x = current_extent_x*(2.0/repeat_0)
                    scatter_scale_y = current_extent_y*(2.0/repeat_1)
                    scatter_axis_sytem.set_scale_x(scatter_scale_x)
                    scatter_axis_sytem.set_scale_y(scatter_scale_y)
                    # Set the translation of the scatter object axis system
                    scatter_axis_sytem.set_translation_x(current_center_x + current_extent_x*((1.0 - repeat_0)/repeat_0) + scatter_scale_x * const_i)
                    scatter_axis_sytem.set_translation_y(current_center_y + current_extent_y*((1.0 - repeat_1)/repeat_1) + scatter_scale_y * const_j)
                    scatter_axis_sytem.set_translation_z(current_center_z + current_extent_z + scatter_extent_z)

                    # Evaluate the local axis system for this object
                    autodiff_matrix = autodiff_variables.evaluate_matrix(scatter_axis_sytem.matrix)                
                    # Set the local matrix of the object to apply the transformation
                    new_scatter_object.matrix_basis = convert_array_to_matrix(autodiff_matrix)
    
    def post_execute(self):
        out = super().post_execute()
        # Focus again on the base object
        focus_on_object(self.inputs["Object"].default_value)
        out["Scattered Objects"] = self.prop_obj_array
        return out
    
    def free(self):
        super().free()
        for object in self.prop_obj_array[1:-1].split(', '):
            try:
                obj = eval(object)
            except:
                log(self.id_data.name, self.name, "free", "Invalid object: " + object, 2)
                continue
            self.id_data.unregister_object(obj)