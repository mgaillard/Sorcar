import bpy
import mathutils
from bpy.props import EnumProperty, PointerProperty, StringProperty
from bpy.types import Node

from ...helper import convert_array_to_matrix
from ...optimization.ScAutodiffVariableCollection import (
    ScAutodiffAxisSystem, ScAutodiffOrientedBoundingBox)
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode


class ScAutodiffLocalTransform(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffLocalTransform"
    bl_label = "Autodiff Local Transform"

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_type: EnumProperty(items=[('TRANSLATION', 'Translation', ''), ('ROTATION', 'Rotation', ''), ('SCALE', 'Scale', '')], update=ScNode.update_value)
    x_default_name: StringProperty(default="", update=ScNode.update_value)
    y_default_name: StringProperty(default="", update=ScNode.update_value)
    z_default_name: StringProperty(default="", update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketString", "Type").init("in_type", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "X").init("x_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Y").init("y_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Z").init("z_default_name", True)

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or (not self.inputs["Type"].default_value in ['TRANSLATION', 'ROTATION', 'SCALE'])
        )
    
    def functionality(self):
        super().functionality()

        # Rename variables for convenience
        current_object = self.inputs["Object"].default_value
        autodiff_variables = self.prop_nodetree.autodiff_variables

        # Read input values
        x_name = self.inputs["X"].default_value
        y_name = self.inputs["Y"].default_value
        z_name = self.inputs["Z"].default_value
        
        # Get corresponding symbols
        x_symbol = autodiff_variables.get_variable_symbol(x_name)
        y_symbol = autodiff_variables.get_variable_symbol(y_name)
        z_symbol = autodiff_variables.get_variable_symbol(z_name)

        # Defaut value per transform type. For scaling => 1.0, for the rest => 0
        if (self.inputs["Type"].default_value == 'SCALE'):
            default_value = 1.0
        else:
            default_value = 0.0

        # If symbols are not available, replace them with a default value
        if x_symbol is None:
            x_symbol = autodiff_variables.get_temporary_const_variable(default_value)
        if y_symbol is None:
            y_symbol = autodiff_variables.get_temporary_const_variable(default_value)
        if z_symbol is None:
            z_symbol = autodiff_variables.get_temporary_const_variable(default_value)

        # Get the bounding box if it exists, then transforms it
        if "OBB" in current_object:
            box_name = current_object["OBB"]

            T = ScAutodiffAxisSystem.fromDefault()

            if (self.inputs["Type"].default_value == 'TRANSLATION'):
                T.translate(x_symbol, y_symbol, z_symbol)
            elif (self.inputs["Type"].default_value == 'ROTATION'):
                T.rotate(x_symbol, y_symbol, z_symbol)
            elif (self.inputs["Type"].default_value == 'SCALE'):
                T.scale(x_symbol, y_symbol, z_symbol)

            autodiff_variables.get_axis_system(box_name).apply(T)
            
            # Evaluate the local axis system for this object
            autodiff_matrix = autodiff_variables.evaluate_matrix(autodiff_variables.get_axis_system(box_name).matrix)
            # Set the local matrix of the object to apply the transformation
            current_object.matrix_basis = convert_array_to_matrix(autodiff_matrix)
