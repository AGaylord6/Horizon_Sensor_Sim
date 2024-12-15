'''
Interface_script.py
Author: Andrew Gaylord

Maya Python script that simulates earth horizon sensor images
Uses IrishSat's Python Simulated Orbital Library (PySOL) to find orbit
    https://github.com/ND-IrishSat/PySOL
    Must clone to documents/maya/scripts
    The generate_orbit_data function can be imported from there

Rendered images are created in selected project folder -> images
'''

import maya.cmds as mc
import maya.api.OpenMaya as om

# need to import all libraries using 
# C:\Program Files\Autodesk\Maya2025\bin>mayapy -m pip install numpy
import numpy as np
import math
import sys, os
import time
import importlib
import math

# import PySOL in specific order
# must pip install astropy, scipy, h5py, matplotlib, geopandas, geodatasets
import PySOL.wmm as wmm
import PySOL.sol_sim
import PySOL.spacecraft as sp
import PySOL.orb_tools as ot

importlib.reload(PySOL.sol_sim)

# ============== PARAMETERS =====================================

earth_radius = 6378
# oe = [0, earth_radius + 450, 0.0000922, 51, -10, 80]
default_oe = [0, earth_radius + 450, 0.0000922, 90, 90, 0]
# total time to simulate (hours)
total_time = 1.6
# timestep between points (seconds)
timestep = 60
# option to only highlight orbit path with new cams + images
cubes_path_no_cams = False
render_image = False
# whether to render as color cam or ir cam (hides correct group)
IR_cam = False
# how many EHS images to create (roughly, depends on timestep)
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
# sensor width (in mm) and desired FOV
cam_FOV = 70.0
sensor_width = 34.0
# array to store all camera objects created
cam_objects = []
earth_object = "earth"
sun_earth_group = "earth_sun"
ir_earth_group = "earth_IR"
# hide the correct groups based on option selected
if IR_cam:
    mc.hide(sun_earth_group)
    mc.showHidden(ir_earth_group)
else:
    mc.hide(ir_earth_group)
    mc.showHidden(sun_earth_group)
# if only rendering cubes to show path, don't render
if cubes_path_no_cams:
    render_images = False

# ============== FUNCTIONS =====================================

def delete_old():
    '''
    Deletes all leftover "orbit" cubes or "ehs" cams
    '''
    orbit_objects = mc.ls("orbit*")
    if orbit_objects:
        mc.delete(orbit_objects)
    cam_objects = mc.ls("ehs*")
    if cam_objects:
        mc.delete(cam_objects)


def create_gui(default_oe):
    '''
    Creates the graphical interface that allows user to define the orbit
    '''
    window_name = "orbital_elements_window"
    
    if mc.window(window_name, exists=True):
        mc.deleteUI(window_name)
    
    mc.window(window_name, title="Orbital Elements Input", widthHeight=(500, 300))
    mc.columnLayout(adjustableColumn=True)
    
    mc.text(label="Enter Orbital Elements:")
    mc.separator(height=30)

    # True anomaly (degrees)
    true_anomaly_field = mc.floatFieldGrp(label="True Anomaly (degrees):", value1=default_oe[0], columnAlign=(1, "center"), columnWidth=(1, 250))
    
    # Semi-major axis (km)
    semi_major_axis_field = mc.floatFieldGrp(label="Semi-major Axis (km):", value1=default_oe[1], columnAlign=(1, "center"), columnWidth=(1, 250))
    
    # Eccentricity
    eccentricity_field = mc.floatFieldGrp(label="Eccentricity:", value1=default_oe[2], columnAlign=(1, "center"), columnWidth=(1, 250), precision=6)
    
    # Inclination (degrees)
    inclination_field = mc.floatFieldGrp(label="Inclination (degrees):", value1=default_oe[3], columnAlign=(1, "center"), columnWidth=(1, 250))

    # Right Ascension of Ascending Node (degrees)
    ra_ascending_node_field = mc.floatFieldGrp(label="Right Ascension of Ascending Node (degrees):", value1=default_oe[4], columnAlign=(1, "center"), columnWidth=(1, 250))
    
    # Argument of Perigee (degrees)
    arg_perigee_field = mc.floatFieldGrp(label="Argument of Perigee (degrees):", value1=default_oe[5], columnAlign=(1, "center"), columnWidth=(1, 250))

    mc.separator(height=30)
    
    # Submit button to apply the orbital elements
    def on_confirm_clicked(*args):
        oe = [
            mc.floatFieldGrp(true_anomaly_field, q=True, value1=True),
            mc.floatFieldGrp(semi_major_axis_field, q=True, value1=True),
            mc.floatFieldGrp(eccentricity_field, q=True, value1=True),
            mc.floatFieldGrp(inclination_field, q=True, value1=True),
            mc.floatFieldGrp(ra_ascending_node_field, q=True, value1=True),
            mc.floatFieldGrp(arg_perigee_field, q=True, value1=True)
        ]
        print("Orbital Elements:", oe)
        # Close the window after applying the orbital elements
        mc.deleteUI(window_name)
        # Call the main function to draw the render the cameras
        main(oe)
    
    # Confirm button
    mc.button(label="Confirm", command=on_confirm_clicked)
    
    mc.showWindow(window_name)


def orient_towards(source, target, ram):
    '''
    Orient source object towards target using quaternion rotation
        Bases direction of up upon direction of travel (ram)
        Adjusts orientation so the object points towards the horizon (Earth's surface)

    TODO: add more options to specify what tilt our cameras are mounted at instead of calculating ideal every time
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
    print("Normalized direction to face (ram): ", up_dir)

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
    theta_center_to_X = math.acos((earth_radius * .001) / d) 
    # Angle of the line connecting the center to the tangent plane
    tangent_angle = math.pi / 2 - theta_center_to_X  

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


def main(oe):
    '''
    Given orbital elements and the parameters from file header, 
    generate orbital data and draw cameras to simulate EHS readings
    '''
    # get gps data in ecef frame from python orbital simulated library
    # also get ram velocity vector for each step (km/s)
    B_field, gps, ram = PySOL.sol_sim.generate_orbit_data(oe, total_time, timestep, None, False, True, True)
    # B_field, gps = PySOL.sol_sim.get_orbit_data(file_name, generate_GPS)
    gps = gps * .001
    ram = ram * .001

    for i, element in enumerate(gps):
        if cubes_path_no_cams:
            # generate cubes that show orbit
            mc.polyCube(name = "orbit" + str(i))
            mc.move(element[0], element[1], element[2])
            mc.scale(.3,.3,.3)

        # only create a camera object every so often
        if i % pic_interval == 0 and not cubes_path_no_cams:
            # create camera and move to current GPS
            mc.camera(name = "ehs")
            mc.move(element[0], element[1], element[2])

            # get created object (name not setting correctly for some reason)
            last = mc.ls(sl=True)[0]
            # add to our list to render later
            cam_objects.append(last)
            # set the FOV of the camera
            # Convert FOV to focal length
            fov_radians = math.radians(cam_FOV)
            focal_length = sensor_width / (2 * math.tan(fov_radians / 2))
            # Set the focal length for the camera
            cmds.setAttr(f'{last}.focalLength', focal_length)
            
            # orient towards the horizon (with respect to RAM) and point towards horizon
            orient_towards(last, earth_object, ram[i])

    if render_image:
        # render all cameras that we created
        print("render every ", pic_interval, " frames")
        
        # create output directory
        project_path = mc.workspace(query=True, rootDirectory=True)
        output_dir = os.path.join(project_path, "images")
        
        # set arnold renderer and different settings
        mc.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
        mc.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
        mc.setAttr("defaultResolution.width", pic_width)
        mc.setAttr("defaultResolution.height", pic_height)
        
        for i, cam in enumerate(cam_objects):
            # set file name and render for every cam we stored
            render_prefix = os.path.join(output_dir, f"{cam}")
            if IR_cam:
                render_prefix = os.path.join(output_dir, f"{cam}_IR")

            mc.setAttr("defaultRenderGlobals.imageFilePrefix", render_prefix, type="string")
            # render every earth horizon sensor (EHS)
            mc.arnoldRender(camera=cam, render=True)
        
        print("All scenes rendered correctly to project folder->images!")

# delete all cube and cam objects from previous iterations
delete_old()

# create the gui (which calls main when "confirm" button is clicked)
create_gui(default_oe)

