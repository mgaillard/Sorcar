import bpy

from bpy.props import PointerProperty, EnumProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode

class ScAutodiffWorldTransform(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffWorldTransform"
    bl_label = "Autodiff World Transform"

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_type: EnumProperty(items=[('LOCATION', 'Location', ''), ('ROTATION', 'Rotation', ''), ('SCALE', 'Scale', '')], update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketString", "Type").init("in_type", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "X")
        self.inputs.new("ScNodeSocketAutodiffNumber", "Y")
        self.inputs.new("ScNodeSocketAutodiffNumber", "Z")

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or (not self.inputs["Type"].default_value in ['LOCATION', 'ROTATION', 'SCALE'])
            or self.inputs["X"].default_value == ""
            or self.inputs["Y"].default_value == ""
            or self.inputs["Z"].default_value == ""
        )
    
    def functionality(self):
        super().functionality()

        # Rename for convenience
        autodiff_variables = self.prop_nodetree.autodiff_variables

        # Read input values
        x_name = self.inputs["X"].default_value
        x_value = autodiff_variables.get_value(x_name, 0.0)
        x_symbol = autodiff_variables.get_variable(x_name)

        y_name = self.inputs["Y"].default_value
        y_value = autodiff_variables.get_value(y_name, 0.0)
        y_symbol = autodiff_variables.get_variable(y_name)

        z_name = self.inputs["Z"].default_value
        z_value = autodiff_variables.get_value(z_name, 0.0)
        z_symbol = autodiff_variables.get_variable(z_name)

        if (self.inputs["Type"].default_value == 'LOCATION'):
            current_object = self.inputs["Object"].default_value
            # Get the bounding box if it exists
            if "OBB" in current_object:
                box_name = current_object["OBB"]
                # Transform the bounding box
                current_object.location.x = x_value
                current_object.location.y = y_value
                current_object.location.z = z_value
                autodiff_variables.get_box(box_name).set_center_x(x_symbol)
                autodiff_variables.get_box(box_name).set_center_y(y_symbol)
                autodiff_variables.get_box(box_name).set_center_z(z_symbol)


            # self.inputs["Object"].default_value.location = self.inputs["Value"].default_value
        # elif (self.inputs["Type"].default_value == 'ROTATION'):
            # self.inputs["Object"].default_value.rotation_euler = self.inputs["Value"].default_value
        # elif (self.inputs["Type"].default_value == 'SCALE'):
            # self.inputs["Object"].default_value.scale = self.inputs["Value"].default_value