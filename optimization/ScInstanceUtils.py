import bpy

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


def create_N_instances(current_object, N):
    #Create duplicates of the object (and its children)
    deselect_all()
    select_recursive(current_object)       

    instances = []
    for i in range(N):
        bpy.ops.object.duplicate_move_linked()
        sel_roots = [obj for obj in bpy.context.selected_objects if obj.parent is None]
        instances += sel_roots 

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
    

    #self.parent_object = parent
    #self.id_data.register_object(self.parent_object)

    # Hide the input
    
    #deselect_recursive(current_object)

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

