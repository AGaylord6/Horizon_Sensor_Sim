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

def orient_towards(source, target):
    '''
    Orient source object towards target using quaternion rotation
    '''

    # get position of two objects
    source_pos = cmds.xform(source, q=True, ws=True, t=True)
    target_pos = cmds.xform(target, q=True, ws=True, t=True)

    vector = om.MVector(target_pos[0] - source_pos[0],
                        target_pos[1] - source_pos[1],
                        target_pos[2] - source_pos[2]).normalize()

    # define up direction as +y
    # TODO: define up direction as direction of RAM? Depends on OE at this step?
    up_dir = om.MVector(0, 1, 0).normalize()

    # cross product to find right vector
    right = vector ^ up_dir
    right.normalize()

    # recompute true up vector
    up_dir = right ^ vector
    up_dir.normalize()

    # Assuming the forward direction is along the Z-axis
    # forward = om.MVector(0, 1, 0)
    # quat = om.MQuaternion(forward, vectorMVector)

    # quaterion rotation matrix
    quat_matrix = om.MMatrix([right.x, right.y, right.z, 0,
                            up_dir.x, up_dir.y, up_dir.z, 0,
                            -vector.x, -vector.y, -vector.z, 0,
                            0, 0, 0, 1])

    # convert rotation matrix to euler angles
    transform = om.MTransformationMatrix(quat_matrix)
    eulers = transform.rotation(om.MEulerRotation.kXYZ)

    # want to find angle that gives us a tangent ray with respect to the earth
    #   one that intersects the edge of the earth from our current height
    # distance = 6.378 + .450
    # angle = math.asin(6.378 / distance)
    # print("triangle: ", math.degrees(angle))
    # # find angle between ray to horizon and x
    # alpha = math.atan2(target_pos[1] - source_pos[1], target_pos[0] - source_pos[0])
    # print("angle between ray and x: ", math.degrees(alpha))
    # tangent_angle = math.degrees(alpha + angle)
    # print("angle to tanget/horizon: ", tangent_angle)

    # Calculate the vector from the sphere center to the point X 
    dx = source_pos[0] - target_pos[0] 
    dy = source_pos[1] - target_pos[1] 
    dz = source_pos[2] - target_pos[2] 
    # Distance from the center of the sphere to point X 
    d = math.sqrt(dx**2 + dy**2 + dz**2) 
    # Calculate the angle of the line connecting the center to point X 
    theta_center_to_X = math.acos(6.378 / d) 
    # The angle between the vector to X and the tangent plane
    tangent_angle = math.pi / 2 - theta_center_to_X 
    # Convert the angle to degrees 
    angle_degrees = math.degrees(tangent_angle)
    # for some reason, only works with x/y axis
    # need to do from basis of looking down at earth (use prev calculations)
    print("angle between X and tangent: ", angle_degrees)

    new_angle = om.MAngle(eulers.x).asDegrees() + angle_degrees
    # TODO: need to orient so that it faces same direction entire time
    print("new angle: ", new_angle)

    # apply rotation to source
    mc.xform(source, rotation=(new_angle,
                                om.MAngle(eulers.y).asDegrees(),
                                om.MAngle(eulers.z).asDegrees()), worldSpace=True)


# delete all "orbit#" objects from previous iterations
delete_old()

earth_radius = 6378
# oe = [0, earth_radius + 450, 0.0000922, 51, -10, 80]
oe = [0, earth_radius + 450, 0.0000922, 0, 90, 0]
total_time = 1.5
timestep = 60
# how many EHS images to create
pic_count = 20
# how often to space along orbit
pic_interval = total_time * 3600 / timestep / pic_count
pic_width = 24
pic_height = 32
file_name = "orbit1.csv"
store_data = False
generate_GPS = True
render_image = True
# TODO: create option for IR vs sun
    # TODO: automatically hide other group + use correct settings based on it

sat_object = "sat"
cam_object = "EHS"
cam_objects = []
earth_object = "earth"

B_field, gps = PySOL.sol_sim.generate_orbit_data(oe, total_time, timestep, file_name, store_data, generate_GPS)
# B_field, gps = PySOL.sol_sim.get_orbit_data(file_name, generate_GPS)
gps = gps * .001

for i, element in enumerate(gps):
    # mc.polyCube(name = "orbit" + str(i))
    # mc.scale(.3,.3,.3)

    # only create a camera object every so often
    if i % pic_interval == 0:
        # create camera and move to current GPS
        mc.camera(name = "ehs")
        mc.move(element[0], element[1], element[2])
        # get created object (name not setting correctly for some reason)
        last = mc.ls(sl=True)
        # add to our list to render later
        cam_objects.append(last[0])
        # orient towards the horizon
        orient_towards(last[0], earth_object)

if render_image:
    # render all cameras that we created
    print("render every ", pic_interval, " frames")
    
    # create output directory
    project_path = mc.workspace(query=True, rootDirectory=True)
    output_dir = project_path + "\\images"
    
    # set arnold renderer and different settings
    mc.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
    mc.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
    # mc.setAttr("defaultResolution.width", pic_width)
    mc.setAttr("defaultResolution.width", 1024)
    # mc.setAttr("defaultResolution.height", pic_height)
    mc.setAttr("defaultResolution.height", 1024)

    
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






