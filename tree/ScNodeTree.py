import bpy
# import linecache

from bpy.props import BoolProperty, StringProperty
from bpy.types import NodeTree
from ..nodes.constants.ScAutodiffNumber import ScAutodiffNumber
from ..helper import update_each_frame, remove_object
from ..optimization.ScAutodiffVariableCollection import ScAutodiffVariableCollection
from ..debug import log, clear_logs, print_traceback
from ..optimization.ScOrientedBoundingBox import ScOrientedBoundingBox

class ScNodeTree(NodeTree):
    bl_idname = 'ScNodeTree'
    bl_label = 'Sorcar'
    bl_icon = 'MESH_CUBE'

    node = None
    links_hash = 0
    objects = []
    variables = {}
    autodiff_variables = ScAutodiffVariableCollection()

    def update_realtime(self, context):
        if not (update_each_frame in bpy.app.handlers.frame_change_post):
            bpy.app.handlers.frame_change_post.append(update_each_frame)
        return None
    prop_realtime: BoolProperty(name="Realtime", update=update_realtime)
    prop_clear_vars: BoolProperty(name="Clear variables", default=True)

    def register_object(self, obj):
        if (obj not in self.objects):
            log(self.name, None, "register_object", "Register object \""+str(obj)+"\"", 2)
            self.objects.append(obj)
    
    def unregister_object(self, obj):
        if obj in self.objects:
            log(self.name, None, "unregister_object", "Unregister object \""+str(obj)+"\"", 2)
            remove_object(obj)
            self.objects.remove(obj)
    
    def unregister_all_objects(self):
        log(self.name, None, "unregister_all_objects", "Objects="+str(len(self.objects)), 2)
        for obj in self.objects:
            log(self.name, None, "unregister_all_objects", "Object="+str(obj), 3)
            remove_object(obj)
        self.objects = []

    def get_links_hash(self):
        links = self.links
        links_data = []
        for link in links:
            links_data.append((link.from_node.name+":"+link.from_socket.identifier, link.to_node.name+":"+link.to_socket.identifier))
        h = hash(str(links_data))
        log(self.name, None, "get_links_hash", "Hash="+str(h), 3)
        return h
    
    def reset_nodes(self, execute):
        log(self.name, None, "reset_nodes", "Execute="+str(execute), 2)
        if (not self.nodes.get(str(self.node))):
            self.node = None
        for i in self.nodes:
            if (hasattr(i, "reset")):
                i.reset(execute)
    
    def update_links(self):
        log(self.name, None, "update_links", "Links="+str(len(self.links)), 2)
        for i in self.links:
            if not (i.to_socket.bl_rna.name == i.from_socket.bl_rna.name):
                if (i.to_socket.bl_rna.name == "ScNodeSocketArrayPlaceholder"):
                    log(self.name, None, "update_links", "FromNode="+i.from_node+", FromSocket="+i.from_socket+", ToNode="+i.to_node+", ToSocket="+i.to_socket, 3)
                    new_socket = i.to_node.inputs.new(i.from_socket.bl_rna.name, i.from_socket.bl_label)
                    self.links.new(i.from_socket, new_socket)
                    self.links.remove(i)

    def update(self):
        self.update_links()
        links_hash = self.get_links_hash()
        if (not self.links_hash == links_hash):
            self.execute_node()
            self.links_hash = links_hash
        self.reset_nodes(False)
    
    def execute_node(self):
        self.reset_nodes(True)
        n = self.nodes.get(str(self.node))
        if (n):
            if (self.prop_clear_vars):
                self.variables.clear()
                self.autodiff_variables.clear()
            clear_logs()
            self.unregister_all_objects()
            if (hasattr(n, "execute")):
                log(self.name, n.name, msg="BEGIN EXECUTION", level=2, mem='SET')
                try:
                    if (not n.execute()):
                        log(self.name, msg="EXECUTION FAILED", level=2, mem='GET')
                    else:
                        log(self.name, msg="EXECUTION SUCCESSFUL", level=2, mem='GET')
                except:
                    print_traceback()
        else:
            log(self.name, None, "execute_node", "Node not found", 2)

    def set_value(self, node_name="Cube", attr_name="in_size", value=1, refresh=True):
        n = self.nodes.get(node_name)
        if (n):
            log(self.name, node_name, "set_value", "Key="+attr_name+", Value="+repr(value)+", Refresh="+str(refresh))
            setattr(n, attr_name, value)
            if (refresh):
                self.execute_node();
        else:
            log(self.name, None, "set_value", "Node not found")
    
    def set_preview(self, node_name="Cube"):
        n = self.nodes.get(node_name)
        if (n):
            log(self.name, node_name, "set_preview", "Set as preview node")
            self.node = n.name
            self.execute_node();
        else:
            log(self.name, None, "set_preview", "Node not found")
            

    def get_float_properties(self):
        float_properties = {}
        # Iterate over all ScAutodiffNumber nodes of type FLOAT
        for node in self.nodes:
            if type(node) == ScAutodiffNumber:
                float_properties[node.name] = float(node.prop_float)

        log(self.name, None, "get_float_properties", repr(float_properties), level=2)
        return float_properties

    def get_float_properties_bounds(self):
        float_properties_bounds = {}
        # Iterate over all ScAutodiffNumber nodes of type FLOAT
        for node in self.nodes:
            if type(node) == ScAutodiffNumber:
                float_properties_bounds[node.name] = {
                    'bounded': node.prop_bounded,
                    'min': float(node.prop_float_min),
                    'max': float(node.prop_float_max)
                }

        log(self.name, None, "get_float_properties_bounds", repr(float_properties_bounds), level=2)
        return float_properties_bounds


    def set_float_properties(self, float_properties):
        log(self.name, None, "set_float_properties", repr(float_properties), level=2)
        # Deactivate execution of the graph
        current_node = self.node
        self.node = None
        # Iterate over all ScAutodiffNumber nodes of type FLOAT
        for node in self.nodes:
            if type(node) == ScAutodiffNumber:
                if node.name in float_properties:
                    # Update the value in the node
                    self.set_value(node_name=node.name, attr_name="prop_float", value=float_properties[node.name], refresh=False)
        # Reactivate the preview node
        self.set_preview(current_node)
        

    def are_target_boxes_all_autodiff(self, target_bounding_boxes):
        # For all objects generated by the procedural tree
        for obj in self.objects:
            # If the current object is a target
            if obj.name in target_bounding_boxes:
                # Check that it has an autodiff OBB
                if "OBB" not in obj:
                    log(self.name, None, "are_boxes_all_autodiff", "No", level=2)
                    return False
        log(self.name, None, "are_boxes_all_autodiff", "Yes", level=2)
        return True
    

    def get_object_boxes(self):
        bounding_boxes = {}
        for obj in self.objects:
            bounding_boxes[obj.name] = ScOrientedBoundingBox.fromObject(obj)
        
        log(self.name, None, "get_object_boxes", repr(bounding_boxes), level=2)
        return bounding_boxes


    def get_object_autodiff_boxes_names(self):
        bounding_boxes = []
        for obj in self.objects:
            if "OBB" in obj:
                bounding_boxes.append(obj["OBB"])
        
        log(self.name, None, "get_object_autodiff_boxes", repr(bounding_boxes), level=2)
        return bounding_boxes
