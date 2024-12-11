'''
Interface_script.py
Author: Andrew Gaylord

Maya Python script that simulates earth horizon sensor images
Uses IrishSat's Python Simulated Orbital Library (PySOL) to find orbit
'''


import maya.cmds as mc
import maya.api.OpenMaya as om

# can import from documents/maya/scripts
#import Final_script

# need to import all libraries using 
# C:\Program Files\Autodesk\Maya2025\bin>mayapy -m pip install numpy
import numpy as np
import math
import sys, os
import time
import importlib

# import PySOL in specific order
# must pip install astropy, scipy, h5py, matplotlib, geopandas, geodatasets
import PySOL.wmm as wmm
import PySOL.sol_sim
import PySOL.spacecraft as sp
import PySOL.orb_tools as ot

importlib.reload(PySOL.sol_sim)

# add to path variable so that subdirectory modules can be imported
#import sys, os
#sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

def delete_old():
    orbit_objects = mc.ls("orbit*")
    if orbit_objects:
        mc.delete(orbit_objects)
    cam_objects = mc.ls("ehs*")
    if cam_objects:
        mc.delete(cam_objects)
    

# TODO: set up function that creates earth, cam, etc
import math

def orient_towards(source, target, ram):
    '''
    Orient source object towards target using quaternion rotation
        Bases direction of up upon direction of travel (ram)
        Adjusts orientation so the object points towards the horizon (Earth's surface)
    '''

    # Get position of two objects (source and target)
    source_pos = cmds.xform(source, q=True, ws=True, t=True)
    target_pos = cmds.xform(target, q=True, ws=True, t=True)

    # Calculate direction vector from source to target
    vector = om.MVector(target_pos[0] - source_pos[0],
                        target_pos[1] - source_pos[1],
                        target_pos[2] - source_pos[2]).normalize()

    # Set the 'up' direction based on the 'ram' velocity vector
    up_dir = om.MVector(ram[0], ram[1], ram[2]).normalize()
    print("Normalized direction to face: ", up_dir)

    # Cross product to calculate the right vector (perpendicular to forward and up)
    right = vector ^ up_dir
    right.normalize()

    # Recompute the 'up' vector to make sure it's perpendicular to both the 'vector' and 'right'
    up_dir = right ^ vector
    up_dir.normalize()

    # Create the quaternion rotation matrix
    quat_matrix = om.MMatrix([right.x, right.y, right.z, 0,
                              up_dir.x, up_dir.y, up_dir.z, 0,
                              -vector.x, -vector.y, -vector.z, 0,
                              0, 0, 0, 1])

    # Convert the rotation matrix to Euler angles
    transform = om.MTransformationMatrix(quat_matrix)
    eulers = transform.rotation(om.MEulerRotation.kXYZ)

    # Calculate the vector from the center of the Earth to the source position (assumed Earth's radius = 6.378 km)
    dx = source_pos[0]  # Earth's center assumed at (0,0,0)
    dy = source_pos[1]
    dz = source_pos[2]
    d = math.sqrt(dx**2 + dy**2 + dz**2)

    # Calculate the angle from the Earth's center to the object's position
    theta_center_to_X = math.acos(6.378 / d)  # Earth's radius is 6.378 km
    tangent_angle = math.pi / 2 - theta_center_to_X  # Angle of the line connecting the center to the tangent plane

    # Convert the angle to degrees
    angle_degrees = math.degrees(tangent_angle)

    # We now calculate the new angle to make the object face the horizon while still facing the target
    new_angle_x = om.MAngle(eulers.x).asDegrees()

    # We can adjust the angle along the X-axis based on the tangent to the horizon
    new_angle_x += angle_degrees

    # Apply the final rotation to source object based on calculated Euler angles
    angle_y = om.MAngle(eulers.y).asDegrees()
    print("new angle_x: ", new_angle_x)

    # Apply the transformation to the source object
    mc.xform(source, rotation=(new_angle_x,
                                angle_y,
                                om.MAngle(eulers.z).asDegrees()), worldSpace=True)


# delete all "orbit#" objects from previous iterations
delete_old()

earth_radius = 6378
# oe = [0, earth_radius + 450, 0.0000922, 51, -10, 80]
# if inclination is 0, ram doesn't work
oe = [0, earth_radius + 450, 0.0000922, 141, 90, 0]
total_time = 1.6
timestep = 60
file_name = "orbit1.csv"
store_data = False
# option to only highlight orbit path with new cams + images
cubes_path_no_cams = False
render_image = False
# whether to render as color cam or ir cam (hides correct group)
IR_cam = False
# how many EHS images to create
pic_count = 20
# how often to space cams along orbit
pic_interval = int(total_time * 3600 / timestep / pic_count)
if IR_cam:
    pic_width = 24
    pic_height = 32
else:
    # settings for higher quality color cam
    pic_width = 512
    pic_height = 512

sat_object = "sat"
cam_object = "EHS"
cam_objects = []
earth_object = "earth"
sun_earth_group = "earth_sun"
ir_earth_group = "earth_IR"
if IR_cam:
    mc.hide(sun_earth_group)
    mc.showHidden(ir_earth_group)
else:
    mc.hide(ir_earth_group)
    mc.showHidden(sun_earth_group)

# get gps data in ecef frame from python orbital simulated library
# also get ram velocity vector for each step (km/s)
B_field, gps, ram = PySOL.sol_sim.generate_orbit_data(oe, total_time, timestep, file_name, store_data, True, True)
# B_field, gps = PySOL.sol_sim.get_orbit_data(file_name, generate_GPS)
gps = gps * .001
ram = ram * .001

# if only rendering cubes to show path, don't create any cams
if cubes_path_no_cams:
    render_images = False
    pic_interval = len(gps) + 1

for i, element in enumerate(gps):
    if cubes_path_no_cams:
        mc.polyCube(name = "orbit" + str(i))
        mc.move(element[0], element[1], element[2])
        mc.scale(.3,.3,.3)

    # only create a camera object every so often
    if i % pic_interval == 0:
        # create camera and move to current GPS
        mc.camera(name = "ehs")
        mc.move(element[0], element[1], element[2])
        # get created object (name not setting correctly for some reason)
        last = mc.ls(sl=True)
        # add to our list to render later
        cam_objects.append(last[0])
        # orient towards the horizon (with respect to RAM)
        orient_towards(last[0], earth_object, ram[i])

if render_image:
    # render all cameras that we created
    print("render every ", pic_interval, " frames")
    
    # create output directory
    project_path = mc.workspace(query=True, rootDirectory=True)
    output_dir = project_path + "\\images"
    
    # set arnold renderer and different settings
    mc.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
    mc.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
    mc.setAttr("defaultResolution.width", pic_width)
    mc.setAttr("defaultResolution.height", pic_height)
    
    for cam in cam_objects:
        # set file name and render for every cam we stored
        mc.setAttr("defaultRenderGlobals.imageFilePrefix", f"{output_dir}/{cam}", type="string")
    
        mc.arnoldRender(camera=cam, render=True)

# move the camera to the last cube (most recently selected)
# last = mc.ls(sl=True)
# cam_pos = mc.xform(last, q=True, ws=True, t=True)
# mc.xform(cam_object, ws=True, t=cam_pos)
# delete last cube
# mc.delete()

# orient the cam to point towards the earth
# orient_towards(cam_object, earth_object)






