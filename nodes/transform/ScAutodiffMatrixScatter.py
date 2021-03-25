import bpy

import mathutils
from bpy.props import PointerProperty, EnumProperty, StringProperty
from bpy.types import Node
from .._base.node_base import ScNode
from .._base.node_operator import ScObjectOperatorNode
from ...helper import convert_array_to_matrix
from ...optimization.ScAutodiffVariableCollection import ScAutodiffOrientedBoundingBox, ScAutodiffAxisSystem

from ...optimization import ScInstanceUtils as instance_utils

import numpy as np
import casadi

class ScAutodiffMatrixScatter(Node, ScObjectOperatorNode):
    bl_idname = "ScAutodiffMatrixScatter"
    bl_label = "Autodiff Matrix Scatter"

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)    
    stepx_default_name: StringProperty(default="", update=ScNode.update_value)
    stepy_default_name: StringProperty(default="", update=ScNode.update_value)
    stepz_default_name: StringProperty(default="", update=ScNode.update_value)

    numx_default_name: StringProperty(default="", update=ScNode.update_value)
    numy_default_name: StringProperty(default="", update=ScNode.update_value)
    numz_default_name: StringProperty(default="", update=ScNode.update_value)
    
    node_name: StringProperty(default="", update=ScNode.update_value)

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketAutodiffNumber", "StepX").init("stepx_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "StepY").init("stepy_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "StepZ").init("stepz_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "NumX").init("numx_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "NumY").init("numy_default_name", True)
        self.inputs.new("ScNodeSocketAutodiffNumber", "NumZ").init("numz_default_name", True)                
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
        
        numX_symbol = casadi.floor(dvars["NumX"]["symbol"])
        numY_symbol = casadi.floor(dvars["NumY"]["symbol"])
        numZ_symbol = casadi.floor(dvars["NumZ"]["symbol"])
        N_symbol = numX_symbol * numY_symbol * numZ_symbol

        

        N = int(autodiff_variables.evaluate_value(N_symbol))
        NumX = int(autodiff_variables.evaluate_value(numX_symbol))
        NumY = int(autodiff_variables.evaluate_value(numY_symbol))
        NumZ = int(autodiff_variables.evaluate_value(numZ_symbol))


        #####################
        # 1. Create instances
        #####################

        self.instances = instance_utils.create_N_instances(autodiff_variables, current_object, N)
        self.parent_object = instance_utils.create_parent_group(
            "MatrixScatterParent" + self.node_name,
            self.instances
        )
        instance_utils.create_empty_bounding_box_for(autodiff_variables, self.parent_object)
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
        
        #2 + 3 = 5 
        matrix_size_X = (numX_symbol - 1) * dvars["StepX"]["symbol"] + numX_symbol * orig_box.get_extent_x()*2
        matrix_size_Y = (numY_symbol - 1) * dvars["StepY"]["symbol"] + numY_symbol * orig_box.get_extent_y()*2
        matrix_size_Z = (numZ_symbol - 1) * dvars["StepZ"]["symbol"] + numZ_symbol * orig_box.get_extent_z()*2

        for x in range(0, NumX):
            for y in range(0, NumY):
                for z in range(0, NumZ):

                    index = x + y * (NumX) + z * (NumX * NumY) 
                    inst = self.instances[index]
                    # Modify
                    tx = (orig_box.get_extent_x()*2 + dvars["StepX"]["symbol"]) * x - matrix_size_X * 0.5 + orig_box.get_extent_x()*2 * 0.5
                    ty = (orig_box.get_extent_y()*2 + dvars["StepY"]["symbol"]) * y - matrix_size_Y * 0.5 + orig_box.get_extent_y()*2 * 0.5
                    tz = (orig_box.get_extent_z()*2 + dvars["StepZ"]["symbol"]) * z - matrix_size_Z * 0.5 + orig_box.get_extent_z() *2* 0.5


                    translate = ScAutodiffAxisSystem.fromDefault()
                    translate.translate(tx,ty,tz)

                    T = translate

                    autodiff_variables.get_axis_system(inst["OBB"]).apply(T)

                    Treal = convert_array_to_matrix(autodiff_variables.evaluate_matrix(T.matrix))
                    inst.matrix_local = Treal @ inst.matrix_local


        


    def post_execute(self):
        out = super().post_execute()
        out["Object"] = self.parent_object        
        return out


        
