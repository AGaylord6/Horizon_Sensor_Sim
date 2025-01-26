'''
Interface_script.py
Author: Andrew Gaylord

Maya Python script that simulates earth horizon sensor images taken from a satellite in Low Earth Orbit (LEO)

Uses IrishSat's Python Simulated Orbital Library (PySOL) to find orbit
    https://github.com/ND-IrishSat/PySOL
    Must clone to documents/maya/scripts
    The generate_orbit_data function can be imported from there

Rendered images are created in selected project folder -> images
'''

import maya.cmds as mc
import maya.api.OpenMaya as om

# need to import all libraries (to maya) using 
# C:\Program Files\Autodesk\Maya2025\bin>.\mayapy -m pip install numpy
import numpy as np
import math
import sys, os
import time
import importlib
import math

from Horizon_Sensor_Sim.params import *
print("INTERVAL: ", pic_interval)
print("IDEAL: ", IDEAL_TILT)
from Horizon_Sensor_Sim.Simulator.magnetorquer import Magnetorquer
from Horizon_Sensor_Sim.Simulator.sat_model import Magnetorquer_Sat
from Horizon_Sensor_Sim.Simulator.simulator import *

# import PySOL in specific order
# must pip install astropy, scipy, h5py, matplotlib, geopandas, geodatasets
import PySOL.wmm as wmm
import PySOL.sol_sim
import PySOL.spacecraft as sp
import PySOL.orb_tools as ot

importlib.reload(PySOL.sol_sim)

# ============== PARAMETERS =====================================

# array to store all camera objects created
cam_objects = []

# hide the correct groups based on option selected
if IR_cam:
    mc.hide(sun_earth_group)
    mc.showHidden(ir_earth_group)
else:
    mc.hide(ir_earth_group)
    mc.showHidden(sun_earth_group)
# if only rendering cubes to show path, don't render
# if cubes_path_no_cams:
    # render_images = False

# if SIMULATING: # TODO
    # render_images = False
    # ideal = False
    # set pic_interval to every time step?

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


def orient_towards(source, target, ram, second=None):
    '''
    Orient source object towards target using quaternion rotation
        Bases direction of up upon direction of travel (ram)
        Adjusts orientation so the object points towards the horizon (Earth's surface)

    TODO: add more options to specify what tilt our cameras are mounted at instead of calculating ideal every time
    '''

    if not IDEAL_TILT:
        tilt = np.random.normal(0, .5, 9)
    else:
        tilt = np.zeros((9))

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

    # Create the quaternion rotation matrix for first cam (adding random tilt)
    quat_matrix = om.MMatrix([right.x + tilt[0], right.y + tilt[1], right.z + tilt[2], 0,
                              up_dir.x + tilt[3], up_dir.y + tilt[4], up_dir.z + tilt[5], 0,
                              -vector.x + tilt[6], -vector.y + tilt[7], -vector.z + tilt[8], 0,
                              0, 0, 0, 1])

    # Convert the rotation matrix to Euler angles
    transform = om.MTransformationMatrix(quat_matrix)
    eulers = transform.rotation(om.MEulerRotation.kXYZ)

    # Calculate the vector from the center of the Earth to the source position (assumed Earth's radius = 6.378 km)
    dx = source_pos[0]  # Earth's center assumed at (0,0,0)
    dy = source_pos[1]
    dz = source_pos[2]
    d = math.sqrt(dx**2 + dy**2 + dz**2)

    if not cam_mount_angle:
        # Calculate the angle from the Earth's center to the object's position
        theta_center_to_X = math.acos((EARTH_RADIUS * .001) / d) 
        # Angle of the line connecting the center to the tangent plane
        tangent_angle = math.pi / 2 - theta_center_to_X  
    
        # Convert the angle to the tangent line to degrees
        # this is the ideal angle our cams would be pointed to directly point at horizon
        angle_degrees = math.degrees(tangent_angle)
    else:
        # record the angle that our cams are mounted at
        angle_degrees = cam_mount_angle

    # adjust the x axis so that the object faces the horizon while still facing ram
    new_angle_x = om.MAngle(eulers.x).asDegrees() + angle_degrees
    print("new angle_x: ", new_angle_x)

    # Apply the transformation to the source object
    mc.xform(source, rotation=(new_angle_x,
                                om.MAngle(eulers.y).asDegrees(),
                                om.MAngle(eulers.z).asDegrees()), worldSpace=True)

    if second:
        # set up direction facing opposite direction as ram
        up_dir = om.MVector(-ram[0], -ram[1], -ram[2]).normalize()
        right = vector ^ up_dir
        right.normalize()
        up_dir = right ^ vector
        up_dir.normalize()
        # quat matrix for second cam (subtracting the tilt so it goes opposite way)
        quat_matrix_second = om.MMatrix([right.x - tilt[0], right.y - tilt[1], right.z - tilt[2], 0,
                                up_dir.x - tilt[3], up_dir.y - tilt[4], up_dir.z - tilt[5], 0,
                                -vector.x + tilt[6], -vector.y + tilt[7], -vector.z + tilt[8], 0,
                                0, 0, 0, 1])
    
        # Convert the rotation matrix to Euler angles
        transform = om.MTransformationMatrix(quat_matrix_second)
        eulers = transform.rotation(om.MEulerRotation.kXYZ)
        # add horizon angle for second cam
        new_angle_x = om.MAngle(eulers.x).asDegrees() + angle_degrees
    
        mc.xform(second, rotation=(new_angle_x,
                                    om.MAngle(eulers.y).asDegrees(),
                                    om.MAngle(eulers.z).asDegrees()), worldSpace=True)


def set_cam_fov(cam, horizontal_fov, vertical_fov):
    '''
    Set cam object to have the specified field of view and sensor size
    '''
    aspect_ratio = pic_width / pic_height

    # set film back -> aperature to sensor size of camera
    mc.setAttr(f'{cam}.horizontalFilmAperture', sensor_width / 25.4)
    mc.setAttr(f'{cam}.verticalFilmAperture', sensor_height / 25.4)

    # Convert FOV to focal length
    fov_radians = math.radians(cam_FOV_vertical)
    focal_length = sensor_height / (2 * math.tan(fov_radians / 2))
    # hardcode focal length that gives 110 angle of view (which I think is for entire square then cut short by resolution)
    focal_length = 9.033
    
    # Set the focal length for the camera
    mc.setAttr(f'{cam}.focalLength', focal_length)


def main(oe):
    '''
    Given orbital elements and the parameters from file header, 
    generate orbital data and draw cameras to simulate EHS readings
    '''
    # get gps data in ecef frame from python orbital simulated library
    # also get ram velocity vector for each step (km/s)
    # TODO: add ram to get_orbit_data
    print("TEST")
    B_earth, gps, ram = PySOL.sol_sim.generate_orbit_data(oe, HOURS, DT, None, False, True, True)
    # B_eart, gps = PySOL.sol_sim.get_orbit_data(B_FIELD_CSV_FILE, true)
    # convert to km
    gps = gps * .001
    ram = ram * .001

    if SIMULATING:
        # create 3 Magnetorquer objects to store in Magnetorquer_Sat object
        mag1 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
        mag2 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
        mag3 = Magnetorquer(n = AIR_NUM_TURNS, area = AIR_AREA, k = K, epsilon = 1)
        mag_array = np.array([mag1, mag2, mag3])

        # initialize object to hold satellite properties
        mag_sat = Magnetorquer_Sat(CUBESAT_BODY_INERTIA, mag_array, VELOCITY_INITIAL, CONSTANT_B_FIELD_MAG, np.array([0.0, 0.0, 0.0]), DT, GYRO_WORKING)

        # run simulation from simulator.py and generate pdf report of results
        sim = Simulator(mag_sat, B_earth)

    for i, element in enumerate(gps):

        # protocal that replaces run_b_dot_sim for ehs simulator
        if SIMULATING:
            ideal_state = sim.find_ideal(i)

            # generate fake sensor data in body frame based on last state
            sim.generateData_step(ideal_state, i)

            current_state = sim.propagate_step(i)

            # calculate total power usage for this time step (Watts)
            # sim.totalPower[i] = sim.power_output[i][0] + sim.power_output[i][1] + sim.power_output[i][2]

            # threshold 0.5-1 degress per second per axis
            # thresholdLow = 0
            # thresholdHigh = DETUMBLE_THRESHOLD

            # angularX = abs(sim.states[i][4])
            # angularY = abs(sim.states[i][5])
            # angularZ = abs(sim.states[i][6])

            # if(sim.finishedTime == -1):
            #     if (thresholdLow <= angularX <= thresholdHigh) and (thresholdLow <= angularY <= thresholdHigh) and (thresholdLow <= angularZ <= thresholdHigh):
            #         # record first time we hit "detumbled" threshold (seconds)
            #         sim.finishedTime = i*DT
                    
            #         # When the "detumbled" threshold is hit, calculate total Energy
            #         # Total Energy is calculated as a "Rieman Sum" of the total power used at each time step multiplied by the time step
            #         for step in range(i):
            #             sim.energy = sim.energy + sim.totalPower[step]*sim.dt

        if cubes_path_no_cams:
            # generate cubes that show orbit
            mc.polyCube(name = "orbit" + str(i))
            mc.move(element[0], element[1], element[2])
            mc.scale(.3,.3,.3)

        # only create a camera object every so often
        if i % pic_interval == 0 and not cubes_path_no_cams:
            # direction that our cam should be oriented
            if IDEAL_TILT:
                direction = ram[i]
            else:
                # to simulate non-ram pointing, pick a random direction to orient ourselves towards
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                direction = ram[i]

            # create camera and move to current GPS
            mc.camera(name = "ehs")
            mc.move(element[0], element[1], element[2])

            # get created object (name not setting correctly for some reason)
            first_cam = mc.ls(sl=True)[0]
            # add to our list to render later
            cam_objects.append(first_cam)
            # set the FOV of the camera
            set_cam_fov(first_cam, cam_FOV_horizontal, cam_FOV_vertical)

            if two_cams:
                # create second cam
                mc.camera(name = "ehs")
                mc.move(element[0], element[1], element[2])
                second_cam = mc.ls(sl=True)[0]
                # add to our list to render later
                cam_objects.append(second_cam)
                set_cam_fov(second_cam, cam_FOV_horizontal, cam_FOV_vertical)
                
                # orient towards the horizon (with respect to RAM) and point towards horizon
                orient_towards(first_cam, earth_object, direction, second_cam)
            else:
                orient_towards(first_cam, earth_object, direction)


    if render_images:
        # render all cameras that we created
        print("render every ", pic_interval, " frames")
        
        # create output directory
        project_path = mc.workspace(query=True, rootDirectory=True)
        output_dir = os.path.join(project_path, "images")
        
        # set arnold renderer and different settings
        mc.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
        mc.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
        mc.setAttr("defaultResolution.width", max(pic_width, pic_height))
        mc.setAttr("defaultResolution.height", max(pic_width, pic_height))
        
        for i, cam in enumerate(cam_objects):
            # set file name and render for every cam we stored
            render_prefix = os.path.join(output_dir, f"{cam}")
            if IR_cam:
                render_prefix = os.path.join(output_dir, f"{cam}_IR")

            if pic_height != pic_width:
                # crop our image to specified resolution
                mc.setAttr("defaultRenderGlobals.useRenderRegion", 1)
                # change sides to make 24 pixels horizontally (hopefully 70 FOV)
                mc.setAttr("defaultRenderGlobals.leftRegion", (pic_height-pic_width) / 2)
                mc.setAttr("defaultRenderGlobals.rightRegion", pic_height - (pic_height-pic_width) / 2 - 1)
                # don't change top/bottom to keep 110 FOV
                mc.setAttr("defaultRenderGlobals.bottomRegion", 0)
                mc.setAttr("defaultRenderGlobals.topRegion", pic_height-1)

            mc.setAttr("defaultRenderGlobals.imageFilePrefix", render_prefix, type="string")
            # render every earth horizon sensor (EHS)
            mc.arnoldRender(camera=cam, render=True)
        
        print("All scenes rendered correctly to project folder->images!")

# delete all cube and cam objects from previous iterations
delete_old()

# create the gui (which calls main when "confirm" button is clicked)
create_gui(ORBITAL_ELEMENTS)

