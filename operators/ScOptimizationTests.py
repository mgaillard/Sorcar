import bpy

from bpy.types import Operator
from ..helper import sc_poll_op
from ..debug import log

from ..experiments.Functions import generate_rosen
from ..experiments.Optimizer import Optimizer

class ScOptimizationTests(Operator):
    """Launch optimization tests"""
    bl_idname = "sorcar.optimization_tests"
    bl_label = "Launch optimization tests"

    @classmethod
    def poll(cls, context):
        return sc_poll_op(context)

    def execute(self, context):
        log("OPERATOR", None, self.bl_idname, "This is a test", 1)

        function = generate_rosen()
        optimizer = Optimizer(function['function'],
                              function['bounds'],
                              function['starting_point'])
        optimizer.optimize(400)
        print('Total optimization time: {} s'.format(optimizer.get_total_time()))

        return {'FINISHED'}