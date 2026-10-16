"""
Render a program graph (local graph JSON) in Blender.
Usage: blender --python tools/blender/program_graph.py -- <graph_local_json_file>
"""
import json
import sys

# Blender's Python API
import bpy
import mathutils

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"

file_path = argv[0]
print(f"Filename: {file_path}")


# Load Program Graph 
with open(file_path, 'r') as file:
    graph_data = json.load(file)

# Reorder the node centers from (z, y, x) to Blender's (x, y, z)
for node in graph_data["node"]:
    node["center"] = (node["center"][2], node["center"][1], node["center"][0])


# Disable color management
bpy.context.scene.view_settings.view_transform = 'Standard'
bpy.context.scene.display_settings.display_device = 'sRGB'

# Parameters for visualization
NODE_RADIUS = 0.8  # Radius of the node spheres
EDGE_THICKNESS = 0.05  # Thickness of the edge cylinders
alpha = 0.01
COLORS = {
    0: (139,69,19, alpha), # saddle brown
    1: (255,20,147, alpha), # deep pink
    2: (255,255,0, alpha), # yellow
    3: (0,128,0, alpha), # green
    4: (0,0,205, alpha), # medium blue
    5: (255,140,0, alpha) # dark orange
}

def create_sphere(location, radius, color, node_id, collection):
    '''Create a sphere at the given location with the specified radius and color.'''
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.object
    obj.name = node_id
    mat = bpy.data.materials.new(name="NodeMaterial")
    mat.diffuse_color = color
    obj.data.materials.append(mat)

    # Link the object to the specified collection
    collection.objects.link(obj)
    bpy.context.collection.objects.unlink(obj)  # Remove from the default collection

    return obj

def create_cylinder(start, end, thickness, color, edge_id, collection):
    '''Create a cylinder between two points with the specified thickness and color'''
    direction = end - start
    length = direction.length
    midpoint = (start + end) / 2
    direction.normalize()

    # Create the cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=thickness, depth=length, location=midpoint)
    obj = bpy.context.object
    obj.name = edge_id

    # Align the cylinder to the direction vector
    z_axis = mathutils.Vector((0, 0, 1))
    rotation = z_axis.rotation_difference(direction).to_euler()
    obj.rotation_euler = rotation

    # Assign material
    mat = bpy.data.materials.new(name="EdgeMaterial")
    mat.diffuse_color = color
    obj.data.materials.append(mat)

    # Link the object to the specified collection
    collection.objects.link(obj)
    bpy.context.collection.objects.unlink(obj)  # Remove from the default collection

    return obj

def clear_scene():
    '''Clear all objects from the current Blender scene'''
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_node_list(nodes):
    # "node_id=Floor_Type_ID": {"position": , "floor": , "type": , "id": , "neighbors": }
    node_dict = {}

    for node in nodes:
        node_id = f"{node['floor']}{node['type']}{node['type_id']}"
        neighbors = [f"{neighbor[0]}{neighbor[1]}{neighbor[2]}" for neighbor in node['neighbors']]

        node_dict[node_id] = {
            "position": node['center'],
            "floor": node['floor'],
            "type": node['type'],
            "type_id": node['type_id'],
            "neighbors": neighbors
        }

    return node_dict

def create_large_plane(size=100):
    '''Create a large plane in the scene.'''
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "GroundPlane"
    # Optionally assign a material to the plane
    mat = bpy.data.materials.new(name="PlaneMaterial")
    mat.diffuse_color = (0.8, 0.8, 0.8, 1)  # Light gray color
    plane.data.materials.append(mat)

def main():
    # Clear the scene
    clear_scene()
    create_large_plane(size=1000)

    # Extract nodes
    nodes = graph_data["node"]

    # "node_id=Floor_Type_ID": {"position": X, "floor": X, "type": X, "id": X, "neighbors": X}
    node_dict = create_node_list(nodes)

    # Create collections for each floor
    floor_collections = {}
    for node_idx in node_dict:
        floor = node_dict[node_idx]["floor"]
        if floor not in floor_collections:
            # Create a new collection for this floor
            floor_collection = bpy.data.collections.new(name=f"Floor_{floor}")
            bpy.context.scene.collection.children.link(floor_collection)
            floor_collections[floor] = floor_collection

    # Create nodes
    for node_idx in node_dict:
        # Get the current collection
        collection = floor_collections[node_dict[node_idx]["floor"]]
        # Create the node
        node_center = node_dict[node_idx]["position"] 
        node_center = tuple(value + 5 for value in node_center)
        node_id = f"node_{node_idx}"
        node_type = node_dict[node_idx]["type"]
        color = COLORS.get(node_type, (1, 1, 1, 1))  # Default to white if type not found
        create_sphere(node_center, NODE_RADIUS, color, node_id, collection)

        # Create edges to neighbors
        for neighbor_idx in node_dict[node_idx]["neighbors"]:
            neighbor_center = node_dict[neighbor_idx]["position"] 
            neighbor_center = tuple(value + 5 for value in neighbor_center)
            edge_id = f"edge_{node_idx}_{neighbor_idx}"
            start_vec = mathutils.Vector(node_center)
            end_vec = mathutils.Vector(neighbor_center)
            create_cylinder(start_vec, end_vec, EDGE_THICKNESS, (1, 1, 1, 1), edge_id, collection)  # Gray for edges

if __name__ == "__main__":
    main()
