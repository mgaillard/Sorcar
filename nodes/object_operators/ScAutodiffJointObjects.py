import bpy

from bpy.props import PointerProperty, EnumProperty, BoolProperty, FloatProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import focus_on_object, convert_array_to_matrix

class ScAutodiffJointObjects(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffJointObjects"
    bl_label = "Autodiff Joint Objects"
    bl_icon = 'MARKER_HLT'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_offset: FloatProperty(name="Offset", update=ScNode.update_value)
    in_axis: EnumProperty(items=[('+X', '+X', ''),
                                 ('+Y', '+Y', ''),
                                 ('+Z', '+Z', ''),
                                 ('-X', '-X', ''),
                                 ('-Y', '-Y', ''),
                                 ('-Z', '-Z', '')], update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Parent")
        self.inputs.new("ScNodeSocketNumber", "Offset").init("in_offset", True)
        self.inputs.new("ScNodeSocketString", "Axis").init("in_axis", True)

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")

    def error_condition(self):
        return (
            super().error_condition()
            or self.inputs["Parent"].default_value == None
            or (not self.inputs["Axis"].default_value in ['+X', '+Y', '+Z', '-X', '-Y', '-Z'])
        )

    def functionality(self):
        super().functionality()

        # Rename variables for convenience
        axis = self.inputs["Axis"].default_value
        current_object = self.inputs["Object"].default_value
        parent_object = self.inputs["Parent"].default_value
        autodiff_variables = self.prop_nodetree.autodiff_variables

        # Parent the two objects
        parent_object.select_set(state=True)
        bpy.context.view_layer.objects.active = parent_object
        bpy.ops.object.parent_set(type = 'OBJECT')

        # Move the child object
        if ("OBB" in parent_object) and ("OBB" in current_object):
            # Compute symbol of the top coordinate of the parent bounding box
            parent_box_name = parent_object["OBB"]
            parent_box = autodiff_variables.get_box(parent_box_name)
            parent_extent_x = parent_box.get_extent_x()
            parent_extent_y = parent_box.get_extent_y()
            parent_extent_z = parent_box.get_extent_z()
            # Compute symbol of the bottom coordinate of the current bounding box
            current_box_name = current_object["OBB"]
            current_box = autodiff_variables.get_box(current_box_name)
            current_extent_x = current_box.get_extent_x()
            current_extent_y = current_box.get_extent_y()
            current_extent_z = current_box.get_extent_z()
            # Convert the float offset to a constant symbol
            offset_symbol = autodiff_variables.get_temporary_const_variable(self.in_offset)
            # Set the translation of the current object axis system
            current_axis_sytem = autodiff_variables.get_axis_system(current_box_name)
            if axis == '+X':
                current_axis_sytem.set_translation_x(parent_extent_x + current_extent_x + offset_symbol)
            elif axis == '-X':
                current_axis_sytem.set_translation_x(-parent_extent_x - current_extent_x - offset_symbol)
            elif axis == '+Y':
                current_axis_sytem.set_translation_y(parent_extent_y + current_extent_y + offset_symbol)
            elif axis == '-Y':
                current_axis_sytem.set_translation_y(-parent_extent_y - current_extent_y - offset_symbol)
            elif axis == '+Z':
                current_axis_sytem.set_translation_z(parent_extent_z + current_extent_z + offset_symbol)
            elif axis == '-Z':
                current_axis_sytem.set_translation_z(-parent_extent_z - current_extent_z - offset_symbol)
            # Evaluate the local axis system for this object
            autodiff_matrix = autodiff_variables.evaluate_matrix(current_axis_sytem.matrix)                
            # Set the local matrix of the object to apply the transformation
            current_object.matrix_basis = convert_array_to_matrix(autodiff_matrix)

        # Clear inverse of the child
        focus_on_object(current_object)
        bpy.ops.object.parent_clear(type = "CLEAR_INVERSE")
