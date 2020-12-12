import bpy

from bpy.types import Node
from mathutils import Vector
from bpy.props import PointerProperty, BoolProperty
from .._base.node_base import ScNode
from ...helper import focus_on_object, get_parent_recursively, get_children_recursively
from ...debug import log

class ScAutodiffBoundingBoxInfo(Node, ScNode):
    bl_idname = "ScAutodiffBoundingBoxInfo"
    bl_label = "Autodiff Bounding Box Info"
    bl_icon = 'FILE_3D'

    prop_nodetree: PointerProperty(name="NodeTree", type=bpy.types.NodeTree, update=ScNode.update_value)
    in_activate_faces: BoolProperty(default=False, update=ScNode.update_value)
    # TODO: add option for recursive exploration of the scene

    def init(self, context):
        super().init(context)
        self.inputs.new("ScNodeSocketObject", "Object")
        self.inputs.new("ScNodeSocketBool", "Activate faces").init("in_activate_faces")
        self.outputs.new("ScNodeSocketArray", "Vertices")
        self.outputs.new("ScNodeSocketArray", "Edges")
        self.outputs.new("ScNodeSocketArray", "Faces")

    def draw_buttons(self, context, layout):
        super().draw_buttons(context, layout)
        layout.prop(self, "prop_nodetree")
    
    def error_condition(self):
        return (
            super().error_condition()
            or self.prop_nodetree == None
            or self.inputs["Object"].default_value == None
        )

    def pre_execute(self):
        super().pre_execute()
        focus_on_object(self.inputs["Object"].default_value, True)
        bpy.ops.object.mode_set(mode='OBJECT')
    
    def post_execute(self):
        out = super().post_execute()

        # Rename variables for convenience
        input_object = self.inputs["Object"].default_value
        activate_faces = self.inputs["Activate faces"].default_value
        objects = self.prop_nodetree.objects
        autodiff_variables = self.prop_nodetree.autodiff_variables

        vertices = []
        edges = []
        faces = []

        # Iterate over all objects of the scene
        parent_object = get_parent_recursively(input_object)
        objects = get_children_recursively(parent_object)

        # Generate bouding boxes for each object in the hierarchy
        for current_object in objects:
            # Get the bounding box if it exists
            if "OBB" in current_object:
                box_name = current_object["OBB"]
                transformed_box = autodiff_variables.compute_transformed_bounding_box(objects, box_name)
                box_points = transformed_box.list_points_to_match()
                # Compute offset for edges and faces
                offset = len(vertices)
                # Add vertices for the bounding box
                for box_point in box_points:
                    value = autodiff_variables.evaluate_vector(box_point)
                    vertices.append(Vector((value[0], value[1], value[2])))
                # Add edges for the bounding box
                edges = edges + transformed_box.list_box_edges(offset)
                # Add faces only if required by the user via the corresponding property
                if activate_faces:
                    faces = faces + transformed_box.list_box_faces(offset)

        # Output
        out["Vertices"] = str(vertices)
        out["Edges"] = str(edges)
        out["Faces"] = str(faces)

        return out