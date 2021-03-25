import bpy

import mathutils
from bpy.props import PointerProperty, EnumProperty, StringProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import convert_array_to_matrix
from ...optimization.ScAutodiffVariableCollection import ScAutodiffOrientedBoundingBox, ScAutodiffAxisSystem, ScOrientedBoundingBox

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

        self.instances = instance_utils.create_N_instances(autodiff_variables, current_object, N)
        self.parent_object = instance_utils.create_parent_group(
            "RadialScatterParent" + self.node_name,
            self.instances
        )

        instance_utils.create_empty_bounding_box_for(autodiff_variables, self.parent_object)
        instance_utils.register_recursive(self.parent_object, self.id_data)
        instance_utils.hide_recursive(current_object)
        
        #####################
        # 2. Distribute instances
        #####################
        #     
        for index, inst in enumerate(self.instances):
            t = index / float(len(self.instances) - 1) * np.pi * 2          
            
            # Modify            
            rot = ScAutodiffAxisSystem.fromDefault()
            rot.rotate_z(t + dvars["Phase"]["symbol"])

            translate = ScAutodiffAxisSystem.fromDefault()
            translate.translate_x( -1.0 * dvars["Radius"]["symbol"])

            scale = ScAutodiffAxisSystem.fromDefault()
            scale.scale(dvars["Scale"]["symbol"],dvars["Scale"]["symbol"],dvars["Scale"]["symbol"])

            T = scale            
            T.apply(rot)
            T.apply(translate)

            autodiff_variables.get_axis_system(inst["OBB"]).apply(T)

            Treal = convert_array_to_matrix(autodiff_variables.evaluate_matrix(T.matrix))
            inst.matrix_local =  Treal @ inst.matrix_local


    def post_execute(self):
        out = super().post_execute()
        out["Object"] = self.parent_object        
        return out
        
