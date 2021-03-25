import bpy

import mathutils
from bpy.props import PointerProperty, EnumProperty, StringProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import convert_array_to_matrix
from ...optimization.ScAutodiffVariableCollection import ScAutodiffOrientedBoundingBox

from ...optimization import ScInstanceUtils as instance_utils

import numpy as np
import casadi


class ScAutodiffRadialScatter(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffRadialScatter"
    bl_label = "Autodiff Radial Scatter"

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)    
    r_default_name: StringProperty(default="", update=ScNode.update_value)
    s_default_name: StringProperty(default="", update=ScNode.update_value)
    phase_default_name: StringProperty(default="", update=ScNode.update_value)
    n_default_name: StringProperty(default="", update=ScNode.update_value)
    node_name: StringProperty(default="", update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        #self.inputs.new("ScNodeSocketString", "Type").init("in_type", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Radius").init("r_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Scale").init("s_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Phase").init("phase_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "Number").init("n_default_name", True)
        self.node_name = str(id(self))


        

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
        )

    def free(self):
        instance_utils.unregister_recursive(self.parent_object, self.id_data)
        
    
    def functionality(self):
        super().functionality()


        current_object = self.inputs["Object"].default_value


        ## Autodiff setup
        autodiff_variables = self.prop_nodetree.autodiff_variables

        dvars = {}
        for key, socket in self.inputs.items():
            symbol_name = socket.default_value
            symbol = autodiff_variables.get_variable_symbol(symbol_name) 
            temp = False
            if symbol is None:
                symbol = autodiff_variables.get_temporary_const_variable(1.0) #TODO per var default
                temp = True

            dvars[key] = {
                "symbol" : symbol,
                "symbol_name" : symbol_name,
                "value" : autodiff_variables.evaluate_value(symbol),
                "temp" : temp
            }

        N = dvars["Number"]["value"]
        N = int(N)

        
        #####################
        # 1. Create instances
        #####################

        self.instances = instance_utils.create_N_instances(current_object, N)
        self.parent_object = instance_utils.create_parent_group(
            "RadialScatterParent" + self.node_name,
            self.instances
        )
        instance_utils.register_recursive(self.parent_object, self.id_data)
        instance_utils.hide_recursive(current_object)
        
        #####################
        # 2. Distribute instances
        #####################

        orig_box = None

        if "OBB" in current_object:
            box_name = current_object["OBB"]
            orig_box = autodiff_variables.get_box(box_name)
        else: 
            raise Exception("No BB box")
        

        for index, inst in enumerate(self.instances):
            t = index / float(len(self.instances) - 1) * np.pi * 2
            
            inst.select_set(True)            
            bpy.context.view_layer.objects.active = inst

            inst_name = "{}_node{}_instance{}".format(box_name, self.node_name, index)

            # Create unique identifier
            inst["OBB"] = inst_name
            inst.name = inst_name            

            #Setup new bbox and axis system
            inst_box = ScAutodiffOrientedBoundingBox.fromOrientedBoundingBox(autodiff_variables.get_box(box_name))
            autodiff_variables.duplicate_axis_system(box_name, inst["OBB"])
            autodiff_variables.set_box(inst["OBB"], inst_box)
            
            # Modify
            tx = casadi.sin(t + dvars["Phase"]["symbol"]) * dvars["Radius"]["symbol"]
            ty = casadi.cos(t + dvars["Phase"]["symbol"]) * dvars["Radius"]["symbol"]
            s = dvars["Scale"]["symbol"]

            autodiff_variables.get_axis_system(inst["OBB"]).scale(s,s,s)
            autodiff_variables.get_axis_system(inst["OBB"]).translate(tx,ty,0)

            # Evaluate the local axis system for this object
            autodiff_matrix = autodiff_variables.evaluate_matrix(autodiff_variables.get_axis_system(inst["OBB"]).matrix)
            # Set the local matrix of the object to apply the transformation
            inst.matrix_basis = convert_array_to_matrix(autodiff_matrix)

            
            inst.select_set(False)
    
        # Setup parent bounding box
        parent_box = ScAutodiffOrientedBoundingBox.fromCenterAndExtent(
            orig_box.get_center_x(), 
            [dvars["Radius"]["symbol"] * dvars["Scale"]["symbol"],dvars["Radius"]["symbol"] * dvars["Scale"]["symbol"],orig_box.get_extent_z() * dvars["Scale"]["symbol"]]
            )
        autodiff_variables.create_default_axis_system(self.parent_object.name)        
        autodiff_variables.set_box(self.parent_object.name, parent_box)         
        self.parent_object["OBB"] = self.parent_object.name


    def post_execute(self):
        out = super().post_execute()
        out["Object"] = self.parent_object        
        return out
        
