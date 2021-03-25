import bpy

from .ScAutodiffVariableCollection import ScAutodiffOrientedBoundingBox, ScOrientedBoundingBox

def deselect_all():
    bpy.context.view_layer.objects.active = None
    for obj in bpy.context.selected_objects:
        obj.select_set(False)  

def select_recursive(obj):
    obj.select_set(True)
    for child in obj.children:
        select_recursive(child)

def hide_recursive(obj):
    obj.hide_render = True
    obj.hide_viewport = True
    obj.hide_set(True)
    for child in obj.children:
        hide_recursive(child)

def deselect_recursive(obj):
    obj.select_set(False)
    for child in obj.children:
        deselect_recursive(child)


def duplicate_autodiff_vars_recursive(autodiff_variables, obj):
    old_identifer = obj["OBB"]
    new_identifer = obj.name

    #Create a new box from the old one
    autodiff_variables.duplicate_box(old_identifer, new_identifer)        
    autodiff_variables.duplicate_axis_system(old_identifer, new_identifer)

    #Update OBB name, in case obj has .001 etc. suffix
    obj["OBB"] = new_identifer

    for child in obj.children:
        duplicate_autodiff_vars_recursive(autodiff_variables, child)
    

def create_N_instances(autodiff_variables, current_object, N):
    #Create duplicates of the object (and its children)
    deselect_all()
    select_recursive(current_object)       

    instances = []
    for i in range(N):
        bpy.ops.object.duplicate_move_linked()
        sel_roots = [obj for obj in bpy.context.selected_objects if obj.parent is None]
        instances += sel_roots 

    # Duplicate bounding boxes/axes systems
    for inst in instances:
        duplicate_autodiff_vars_recursive(autodiff_variables, inst)


    deselect_all()

    return instances


def register_instances(instances, nodegraph_id_data):
    for inst in instances:
        nodegraph_id_data.register_object(inst)

def register_recursive(obj, nodegraph_id_data):
    nodegraph_id_data.register_object(obj)    
    for child in obj.children:
        register_recursive(child, nodegraph_id_data)

def unregister_recursive(obj, nodegraph_id_data):
    nodegraph_id_data.unregister_object(obj)    
    for child in obj.children:
        unregister_recursive(child, nodegraph_id_data)



def create_parent_group(parent_name, objects):
    # Create empty parent
    parent = bpy.data.objects.new(parent_name, None)        
    parent.empty_display_size = 2
    parent.empty_display_type = 'PLAIN_AXES'   
    bpy.context.scene.collection.objects.link(parent)
   
    # Select instances
    for inst in objects:
        inst.select_set(True)

    # Select and activate parent
    parent.select_set(True)        
    bpy.context.view_layer.objects.active = parent

    # Perform parenting
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)

    #Deselect all
    deselect_all()

    return parent



def create_empty_bounding_box_for(autodiff_variables, obj):
    box = ScOrientedBoundingBox.fromObject(obj) #Try creating from object, might be empty

    autodiff_variables.create_default_axis_system(obj.name)                
    autodiff_variables.set_box_from_constants(obj.name, box)

    obj["OBB"] = obj.name
    
    