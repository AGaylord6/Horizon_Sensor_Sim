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
import cv2

# ensure that our libraries are loaded correctly and recognized on path var
pysol_path = "C:\\Users/agaylord/Documents/maya/scripts\\PySOL"
if pysol_path not in sys.path:
    sys.path.append(pysol_path)
ehs_path = "C:\\Users/agaylord/Documents/maya/scripts\\Horizon_Sensor_Sim" 
if ehs_path not in sys.path:
    sys.path.append(ehs_path)

# reload all library functions while editing to make sure things are updated
import Horizon_Sensor_Sim.params
importlib.reload(Horizon_Sensor_Sim.params)
# importlib.reload(Horizon_Sensor_Sim.Simulator.B_dot)
import Horizon_Sensor_Sim.Simulator.B_dot
importlib.reload(Horizon_Sensor_Sim.Simulator.B_dot)
import Horizon_Sensor_Sim.Simulator.propagate
importlib.reload(Horizon_Sensor_Sim.Simulator.propagate)
# importlib.reload(Horizon_Sensor_Sim.Simulator.graphing)
import Horizon_Sensor_Sim.Simulator.graphing
importlib.reload(Horizon_Sensor_Sim.Simulator.graphing)
# importlib.reload(Horizon_Sensor_Sim.Simulator.saving)
import Horizon_Sensor_Sim.Simulator.saving
importlib.reload(Horizon_Sensor_Sim.Simulator.saving)
import Horizon_Sensor_Sim.Simulator.camera
importlib.reload(Horizon_Sensor_Sim.Simulator.camera)
import Horizon_Sensor_Sim.Simulator.image_processing
importlib.reload(Horizon_Sensor_Sim.Simulator.image_processing)
import Horizon_Sensor_Sim.Simulator.nadir_point
importlib.reload(Horizon_Sensor_Sim.Simulator.nadir_point)
import Horizon_Sensor_Sim.Simulator.BangBang
importlib.reload(Horizon_Sensor_Sim.Simulator.BangBang)

from Horizon_Sensor_Sim.Simulator.magnetorquer import Magnetorquer
from Horizon_Sensor_Sim.Simulator.sat_model import Magnetorquer_Sat
from Horizon_Sensor_Sim.Simulator.simulator import *
importlib.reload(Horizon_Sensor_Sim.Simulator.magnetorquer)
importlib.reload(Horizon_Sensor_Sim.Simulator.sat_model)
importlib.reload(Horizon_Sensor_Sim.Simulator.simulator)

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
if cubes_path_no_cams:
    render_images = False

if SIMULATING: 
    ideal = False
    # TODO: set pic_interval to every time step? this might not do anything here lol

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


def quat_rotate(obj, quat, second=None):
    """
    Apply a quaternion rotation to an object in Maya.
    
    @params:
        obj: cube/cam to be rotated
        quat: quaternion in (w, x, y, z) form
            https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
            error quat: https://stackoverflow.com/questions/23860476/how-to-get-opposite-angle-quaternion
        second (optional): second ehs cam. They will be tilted by cam_mount_angle
    """
    # Convert quaternion to MQuaternion object (x, y, z, w)
    q = om.MQuaternion(quat[1], quat[2], quat[3], quat[0])
    q = q.normal()

    # Convert the quaternion to Euler angles
    euler_rotation = q.asEulerRotation()

    # Convert Euler angles to degrees (Maya uses degrees for rotations)
    euler_rotation_degrees = [angle * (180.0 / 3.14159265359) for angle in euler_rotation]
    if second:
        # define the euler rotation that has the cameras face towards each other
        # TODO: are they oriented on correct face (x, y, etc)? Does it matter for our controls?
        cam2_euler = [-euler_rotation_degrees[0] + 180, -euler_rotation_degrees[1], euler_rotation_degrees[2] + 180]

        # apply camera tilt if we're workin with two cam objects
        euler_rotation_degrees[0] -= (90 - cam_mount_angle)
        cam2_euler[0] -= (90 - cam_mount_angle)
        mc.xform(second, rotation=cam2_euler, worldSpace=True)

    # Apply the rotation to the object
    mc.xform(obj, rotation=euler_rotation_degrees, worldSpace=True)


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
    source_pos = mc.xform(source, q=True, ws=True, t=True)
    target_pos = mc.xform(target, q=True, ws=True, t=True)

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


def create_two_cams(gps, curr_quat, output_dir):
    '''
    Create two earth horizon sensors (EHS) at current orientation
    Render their image and return from proper directory
    @params:
        gps (1x3 array): current gps coordinates of satellite
        curr_quat (1x4 array): current orientation of satellite
        output_dir (string): full path of location to store images in
    @returns:
        image1 (24x32 array): rendered image
        image2 
    '''

    # create camera and move to current GPS
    mc.camera(name = "ehs")
    mc.move(gps[0], gps[1], gps[2])

    # get created object (name not setting correctly for some reason)
    first_cam = mc.ls(sl=True)[0]
    # add to our list to render later
    cam_objects.append(first_cam)
    # set the FOV of the camera
    set_cam_fov(first_cam, cam_FOV_horizontal, cam_FOV_vertical)

    # create second cam
    mc.camera(name = "ehs")
    mc.move(gps[0], gps[1], gps[2])
    second_cam = mc.ls(sl=True)[0]
    # add to our list to render later
    cam_objects.append(second_cam)
    set_cam_fov(second_cam, cam_FOV_horizontal, cam_FOV_vertical)
    
    # orient our cameras towards current orientation and render images
    quat_rotate(first_cam, curr_quat, second_cam)

    if render_images:
    
        # set file name for first cam
        render_prefix = os.path.join(output_dir, f"{first_cam}_IR_first")
        mc.setAttr("defaultRenderGlobals.imageFilePrefix", render_prefix, type="string")
        # render first earth horizon sensor (EHS)
        mc.arnoldRender(camera=first_cam, render=True)
    
        # set file name for second cam
        render_prefix = os.path.join(output_dir, f"{first_cam}_IR_second")
        mc.setAttr("defaultRenderGlobals.imageFilePrefix", render_prefix, type="string")
        # render second earth horizon sensor (EHS)
        mc.arnoldRender(camera=second_cam, render=True)
    
        # fetch our recently rendered images with openCV
        # Construct the absolute path to the image
        image_path = os.path.join(output_dir, f"{first_cam}_IR_first_1.png")
        # read our image
        image1 = cv2.imread(image_path)
        image_path = os.path.join(output_dir, f"{first_cam}_IR_second_1.png")
        image2 = cv2.imread(image_path)
    
        return image1, image2

    else:
        return -1, -1


def main(oe):
    '''
    Given orbital elements and the parameters from file header, 
    generate orbital data and draw cameras to simulate EHS readings
    '''
    # get gps data in ecef frame from python orbital simulated library
    # also get ram velocity vector for each step (km/s)
    # TODO: add ram to get_orbit_data
    if not GENERATE_NEW:
        B_earth, gps = PySOL.sol_sim.get_orbit_data(CSV_FILE, GPS=True)
    else:
        B_earth, gps, ram = PySOL.sol_sim.generate_orbit_data(oe, HOURS, DT, CSV_FILE, store_data=False, GPS=True, RAM=True)
        ram = ram * .001
    if len(B_earth) > int(TF / DT):
        B_earth = B_earth[:int(TF / DT)]
    elif len(B_earth) < int(TF / DT):
        print("ERROR: not enough data points in B_earth. {} needed, {} created".format(int(TF/DT), len(B_earth)))
        return
    # convert to km
    gps = gps * .001
    # initialize current state
    current_state = np.zeros((STATE_SPACE_DIMENSION))
    second_cam = None
    first_cam = None
    # set arnold renderer and different settings
    # find output directory (current project folder -> images)
    project_path = mc.workspace(query=True, rootDirectory=True)
    output_dir = os.path.join(project_path, "images")
    mc.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
    mc.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
    mc.setAttr("defaultResolution.width", max(pic_width, pic_height))
    mc.setAttr("defaultResolution.height", max(pic_width, pic_height))
    
    if pic_height != pic_width:
        # crop our image to specified resolution
        mc.setAttr("defaultRenderGlobals.useRenderRegion", 1)
        # change sides to make 24 pixels horizontally (hopefully 70 FOV)
        mc.setAttr("defaultRenderGlobals.leftRegion", (pic_height-pic_width) / 2)
        mc.setAttr("defaultRenderGlobals.rightRegion", pic_height - (pic_height-pic_width) / 2 - 1)
        # don't change top/bottom to keep 110 FOV
        mc.setAttr("defaultRenderGlobals.bottomRegion", 0)
        mc.setAttr("defaultRenderGlobals.topRegion", pic_height-1)

    if SIMULATING:
        # create 3 Magnetorquer objects to store in Magnetorquer_Sat object
        mag1 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
        mag2 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
        # mag3 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
        mag3 = Magnetorquer(n = AIR_NUM_TURNS, area = AIR_AREA, k = K, epsilon = 1)
        mag_array = np.array([mag1, mag2, mag3])

        # initialize object to hold satellite properties
        mag_sat = Magnetorquer_Sat(CUBESAT_BODY_INERTIA, mag_array, VELOCITY_INITIAL, CONSTANT_B_FIELD_MAG, np.array([0.0, 0.0, 0.0]), DT, GYRO_WORKING)

        # run simulation from simulator.py and generate pdf report of results
        sim = Simulator(mag_sat, B_earth)

    for i, element in enumerate(gps):

        # don't exceed simulation time, even if we have more gps data
        if SIMULATING and i >= sim.n:
            break

        # protocal that replaces run_b_dot_sim for ehs simulator
        if SIMULATING and i != 0:

            # generate ideal state based on last so that we can better estimate sensor data
            ideal_state = sim.find_ideal(i)
            
            # generate fake sensor data in body frame based on ideal guess
            sim.generateData_step(ideal_state, i)

            if not cubes_path_no_cams and i % 2 == 0: # i % pic_interval == 0 and 
                # generate ehs, render image, and fetch from dir
                image1, image2 = create_two_cams(element, ideal_state[:4], output_dir)

                if render_images: 
                    # process our images and store results in mag_sat
                    sim.process_images(image1, image2, i)

            # check what protocol we should be in and update state
            sim.mag_sat.state = sim.check_state(i)

            # decide voltage for self.voltages[i] (depending on state)
            sim.controls(i)
            
            # propagate based on voltages[i]
            current_state = sim.propagate_step(i)
            # print("current state: ", current_state)

            # calculate total power usage for this time step (Watts)
            sim.totalPower[i] = sim.power_output[i][0] + sim.power_output[i][1] + sim.power_output[i][2]

        if cubes_path_no_cams and i % (DT * 3000) == 0: # 200 for dt = .5
            # generate cubes every so often that show orbit
            mc.polyCube(name = "orbit" + str(i))
            mc.move(element[0], element[1], element[2])
            mc.scale(.3,.3,.3)
            if SIMULATING:
                quat_rotate("orbit" + str(i), current_state[:4])
        '''
        # only create a camera object every so often
        if i % pic_interval == 0 and not cubes_path_no_cams:
            # direction that our cam should be oriented
            if IDEAL_TILT:
                direction = ram[i]
            else:
                # to simulate non-ram pointing, pick a random direction to orient ourselves towards
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                # direction = ram[i]

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
                
                if not SIMULATING:
                    # orient towards the horizon (with respect to RAM) and point towards horizon
                    orient_towards(first_cam, earth_object, direction, second_cam)
            
            if not two_cams and not SIMULATING:
                # orient cam towards horizon
                orient_towards(first_cam, earth_object, direction)

            if SIMULATING:
                # orient our cameras towards current orientation and render images
                quat_rotate(first_cam, current_state[:4], second_cam)

                # set file name for first cam
                render_prefix = os.path.join(output_dir, f"{first_cam}_IR_first")
                mc.setAttr("defaultRenderGlobals.imageFilePrefix", render_prefix, type="string")
                # render first earth horizon sensor (EHS)
                mc.arnoldRender(camera=first_cam, render=True)

                # set file name for second cam
                render_prefix = os.path.join(output_dir, f"{first_cam}_IR_second")
                mc.setAttr("defaultRenderGlobals.imageFilePrefix", render_prefix, type="string")
                # render second earth horizon sensor (EHS)
                mc.arnoldRender(camera=second_cam, render=True)
        '''

    if render_images and not SIMULATING:
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

    if SIMULATING:
        sim.plot_and_viz_results()

# delete all cube and cam objects from previous iterations
delete_old()

# create the gui (which calls main when "confirm" button is clicked)
create_gui(ORBITAL_ELEMENTS)

