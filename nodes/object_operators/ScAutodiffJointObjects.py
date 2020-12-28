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

    in_align_0: EnumProperty(items=[('CENTER', 'Center', ''), ('LEFT', 'Left', ''), ('RIGHT', 'Right', '')], update=ScNode.update_value)
    in_align_1: EnumProperty(items=[('CENTER', 'Center', ''), ('LEFT', 'Left', ''), ('RIGHT', 'Right', '')], update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Parent")
        self.inputs.new("ScNodeSocketNumber", "Offset").init("in_offset", True)
        self.inputs.new("ScNodeSocketString", "Axis").init("in_axis", True)
        self.inputs.new("ScNodeSocketString", "Align0").init("in_align_0", True)
        self.inputs.new("ScNodeSocketString", "Align1").init("in_align_1", True)

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")

    def error_condition(self):
        return (
            super().error_condition()
            or self.inputs["Parent"].default_value == None
            or (not self.inputs["Axis"].default_value in ['+X', '+Y', '+Z', '-X', '-Y', '-Z'])
            or (not self.inputs["Align0"].default_value in ['CENTER', 'LEFT', 'RIGHT'])
            or (not self.inputs["Align1"].default_value in ['CENTER', 'LEFT', 'RIGHT'])
        )

    def functionality(self):
        super().functionality()

        # Rename variables for convenience
        axis = self.inputs["Axis"].default_value
        align_0 = self.inputs["Align0"].default_value
        align_1 = self.inputs["Align1"].default_value
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

            # Alignment on the first axis
            if align_0 == 'LEFT':
                align_0_x = -parent_extent_x + current_extent_x
                align_0_y = -parent_extent_y + current_extent_y
                align_0_z = -parent_extent_z + current_extent_z
            elif align_0 == 'RIGHT':
                align_0_x = parent_extent_x - current_extent_x
                align_0_y = parent_extent_y - current_extent_y
                align_0_z = parent_extent_z - current_extent_z
            else: # CENTER
                align_0_x = 0.0
                align_0_y = 0.0
                align_0_z = 0.0

            # Alignment on the second axis
            if align_1 == 'LEFT':
                align_1_x = -parent_extent_x + current_extent_x
                align_1_y = -parent_extent_y + current_extent_y
                align_1_z = -parent_extent_z + current_extent_z
            elif align_1 == 'RIGHT':
                align_1_x = parent_extent_x - current_extent_x
                align_1_y = parent_extent_y - current_extent_y
                align_1_z = parent_extent_z - current_extent_z
            else: # CENTER
                align_1_x = 0.0
                align_1_y = 0.0
                align_1_z = 0.0

            # Set the translation of the current object axis system
            current_axis_sytem = autodiff_variables.get_axis_system(current_box_name)
            if axis == '+X':
                current_axis_sytem.set_translation_x(parent_extent_x + current_extent_x + offset_symbol)
                current_axis_sytem.set_translation_y(align_0_y)
                current_axis_sytem.set_translation_z(align_1_z)
            elif axis == '-X':
                current_axis_sytem.set_translation_x(-parent_extent_x - current_extent_x - offset_symbol)
                current_axis_sytem.set_translation_y(align_0_y)
                current_axis_sytem.set_translation_z(align_1_z)
            elif axis == '+Y':
                current_axis_sytem.set_translation_y(parent_extent_y + current_extent_y + offset_symbol)
                current_axis_sytem.set_translation_z(align_0_z)
                current_axis_sytem.set_translation_x(align_1_x)
            elif axis == '-Y':
                current_axis_sytem.set_translation_y(-parent_extent_y - current_extent_y - offset_symbol)
                current_axis_sytem.set_translation_z(align_0_z)
                current_axis_sytem.set_translation_x(align_1_x)
            elif axis == '+Z':
                current_axis_sytem.set_translation_z(parent_extent_z + current_extent_z + offset_symbol)
                current_axis_sytem.set_translation_x(align_0_x)
                current_axis_sytem.set_translation_y(align_1_y)
            elif axis == '-Z':
                current_axis_sytem.set_translation_z(-parent_extent_z - current_extent_z - offset_symbol)
                current_axis_sytem.set_translation_x(align_0_x)
                current_axis_sytem.set_translation_y(align_1_y)
                
            # Evaluate the local axis system for this object
            autodiff_matrix = autodiff_variables.evaluate_matrix(current_axis_sytem.matrix)                
            # Set the local matrix of the object to apply the transformation
            current_object.matrix_basis = convert_array_to_matrix(autodiff_matrix)

        # Clear inverse of the child
        focus_on_object(current_object)
        bpy.ops.object.parent_clear(type = "CLEAR_INVERSE")
