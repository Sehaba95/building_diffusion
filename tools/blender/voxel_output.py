"""
Render a voxel graph (raw voxel JSON, or a generated output of complete_test) in Blender.
Usage: blender --python tools/blender/voxel_output.py -- <voxel_json_file>
"""
import json
import math
import pathlib
import sys

# Blender's Python API
import bpy
from mathutils import Quaternion

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"

# Building-GAN
def arg_max(list):
    index = -1
    for i in range(len(list)):
        if list[i] > 0:
            return i
    return index

def one_hot_vector(_id):
    number_of_class = 6
    assert _id < number_of_class
    return [1 if i == _id else 0 for i in range(number_of_class)]

def process_voxel_graph(voxel_graph):
    """Voxel floors, features (dimension, coordinate, weight) and one-hot labels, from the raw voxel JSON."""
    dimension_norm_factor = 100
    voxel_floor_index, voxel_feature, voxel_label = [], [], []
    for n in voxel_graph["voxel_node"]:
        voxel_label.append(one_hot_vector(n["type"]))
        voxel_feature.append([*[x / dimension_norm_factor for x in n["dimension"] + n["coordinate"]], n["weight"]])
        voxel_floor_index.append(n["location"][0])

    return {"voxel_floor_index": voxel_floor_index,  # N
            "voxel_feature": voxel_feature,  # N x 7
            "voxel_label": voxel_label,  # N x C
            }

def add_cube(location, dimension, color_material, index, voxel_floor):
    # add a cube into the scene
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, scale=dimension)
    # get a reference to the currently active object
    current_cube = bpy.context.active_object
    
    # Name the voxel
    current_cube.name = "voxel_" + str(index)
    # Give it a color 
    current_cube.data.materials.append(color_material)
    
    # Remove object from all collections not used in a scene
    bpy.ops.collection.objects_remove_all()
    
    bpy.data.collections['Floor_'+str(voxel_floor)].objects.link(current_cube)

def create_floor_collections(number_of_floors):
    
    for floor in range(number_of_floors):
        # New Collection
        my_coll = bpy.data.collections.new("Floor_"+str(floor))

        # Add collection to scene collection
        bpy.context.scene.collection.children.link(my_coll)
    

def create_voxel_building(voxel_graph):
    """
    Prepare data for visualization and add cubes to the scene
    """
    
    number_of_floors = max(voxel_graph["voxel_floor_index"]) + 1 
    create_floor_collections(number_of_floors)
    
    # Alpha: opacity of the color
    alpha = 0.005

    # Prepare color materials for visualization
    brown_color = bpy.data.materials.new("Brown")
    brown_color.diffuse_color = (139,69,19, alpha) # saddle brown

    pink_color = bpy.data.materials.new("Pink") 
    pink_color.diffuse_color = (255,20,147, alpha) # deep pink
    
    yellow_color = bpy.data.materials.new("Yellow")
    yellow_color.diffuse_color = (255,255,0, alpha) # yellow
    
    green_color = bpy.data.materials.new("Green")
    green_color.diffuse_color = (0,128,0, alpha) # green
    
    blue_color = bpy.data.materials.new("Blue") 
    blue_color.diffuse_color = (0,0,205, alpha) # medium blue
    
    orange_color = bpy.data.materials.new("Orange") 
    orange_color.diffuse_color = (255,140,0, alpha) # dark orange

    # Initialize the list of colors: one color for each type of voxels
    colors_materials = [ 
        brown_color, #"brown", 
        pink_color, #"pink", 
        yellow_color, #"yellow", 
        green_color,#"green", 
        blue_color, #"blue", 
        orange_color, #"orange"
        ]

    for i in range(len(voxel_graph["voxel_floor_index"])):
        # Get the floor of the current voxel
        voxel_floor = voxel_graph["voxel_floor_index"][i]

        # Get dimension - coordinate - weight of each voxel
        z_dim, y_dim, x_dim, z_pos, y_pos, x_pos, _ = voxel_graph["voxel_feature"][i]
        # Multiply the dimension and position by 100 to get the real values
        z_dim, y_dim, x_dim, z_pos, y_pos, x_pos = int(100 *z_dim), int(100 *y_dim), int(100 *x_dim), int(100 *z_pos), int(100 *y_pos), int(100 * x_pos)

        # Cube center
        location = (x_pos+x_dim/2, y_pos+y_dim/2, z_pos+z_dim/2)
        dimension = (x_dim, y_dim, z_dim)
        
        # Voxel label (-1: unassigned voxels are not drawn)
        voxel_label = arg_max(voxel_graph["voxel_label"][i])

        if voxel_label != -1:
            # Add the voxel to the scene using location, dimension and give the color based on the type of node
            add_cube(location, dimension, colors_materials[voxel_label], i, voxel_floor)

# From YouTube tutorial    
def delete_all_objects():
    """Removing all of the objects from the scene and remove all collections"""
    # make sure that we are in object mode
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")

    # select all object in the scene
    bpy.ops.object.select_all(action="SELECT")

    # delete all selected objects in the scene
    bpy.ops.object.delete()

def load_data():
    # The voxel JSON file is the first argument after "--"
    path_to_file = argv[0]

    print("="*100)
    print(f"path_to_file: {path_to_file}")
    print("="*100)

    # Check if file exists and read data
    if not pathlib.Path(path_to_file).exists():
        raise FileNotFoundError(f"File not found: {path_to_file}")

    with open(path_to_file, "r") as in_file_obj:
        text = in_file_obj.read()
        data = json.loads(text)

    return data

def set_viewport_view(
    view_location=(0, 0, 0),
    pitch_degrees=0,   # Rotation around X-axis (up/down)
    yaw_degrees=0,     # Rotation around Y-axis (left/right)
    roll_degrees=0     # Rotation around Z-axis (tilt)
):
    """
    Set the 3D Viewport to look from a specific orientation defined by Euler-style angles.
    
    :param view_location: The point in space that the view should focus on
    :param pitch_degrees: Pitch angle (X-axis) - up/down
    :param yaw_degrees:   Yaw angle (Y-axis) - left/right
    :param roll_degrees:  Roll angle (Z-axis) - tilt
    """
    # Convert degrees to radians
    pitch = math.radians(pitch_degrees)
    yaw = math.radians(yaw_degrees)
    roll = math.radians(roll_degrees)

    # Create individual quaternions for each axis
    q_pitch = Quaternion((1, 0, 0), pitch)  # X-axis rotation
    q_yaw = Quaternion((0, 1, 0), yaw)      # Y-axis rotation
    q_roll = Quaternion((0, 0, 1), roll)    # Z-axis rotation

    # Combine rotations: apply roll -> pitch -> yaw (you can change order if needed)
    final_rotation = q_yaw @ q_pitch @ q_roll

    # Apply to 3D view
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    region_3d = space.region_3d
                    region_3d.view_location = view_location
                    region_3d.view_rotation = final_rotation.copy()
                    region_3d.view_distance = 50  # Adjust zoom level
                    return

def main():
    # delete all objects in the scene
    delete_all_objects()

    set_viewport_view(
    view_location=(50, -15, 20),
    pitch_degrees=60,   # Rotation around X-axis (up/down)
    yaw_degrees=0,     # Rotation around Y-axis (left/right)
    roll_degrees=0)    # Rotation around Z-axis (tilt)
    

    # Load the JSON file
    voxel_graph = load_data()

    # Process the raw input voxel JSON
    voxel_graph = process_voxel_graph(voxel_graph)

    # Add voxels to the scene and color each voxel based on its type
    create_voxel_building(voxel_graph)

main()