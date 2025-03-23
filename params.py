'''
params.py
Authors: Andrew Gaylord, Michael Kuczun

Holds all constants/parameters/initial values for our detumbling simulation
'''

import numpy as np
import math

DEGREES = False

# ============  INITIAL VALUES  ======================================

QUAT_INITIAL = np.array([1.0, 1.0, 0.0, 0.0])
# we want to start with 15 degrees/s in each axis
# VELOCITY_INITIAL = np.array([15.0,-10.0,10.0])
VELOCITY_INITIAL = np.array([0.15, 0.0, 0.0])
# convert to rad/s
if not DEGREES:
    VELOCITY_INITIAL *= math.pi / 180
VOLTAGES_INITIAL = np.array([0.0, 0.0, 0.0])
CURRENTS_INITIAL = np.array([0.0, 0.0, 0.0])
RW_INITIAL = np.array([0.0, 0.0, 0.0, 0.0])
STARTING_PROTOCOL = "search" # "detumble", "search", "point"

# ============  ORBITAL DYNAMICS  ==================================================

EARTH_RADIUS = 6378

# see generate_orbit_data in sol_sim.py for more 
# this gives us near-polar, sun-synchronous LEO orbit (according to chatGPT)
# ORBITAL_ELEMENTS = np.array([0, 6800, 0.0000922, 97.5, 150, 0])
ORBITAL_ELEMENTS = [90, EARTH_RADIUS + 450, 0.0000922, 90, 90, 0]

# standard gravitational parameter for earth (m^3/s^2)
GRAVITY_EARTH = 3.986004418e14

# time for one orbit from function (hours) 
TIME_PER_ORBIT = 1.550138888888889 # 1.525 from trial/error
# orbital period (time to complete one orbit) according to Kepler's 3rd law (seconds)
ORBITAL_PERIOD = 2 * np.pi * math.sqrt((ORBITAL_ELEMENTS[1] * 1000)**3 / GRAVITY_EARTH)
# print("Orbital period: ", ORBITAL_PERIOD / 3600, "hours per orbit")

# Our North West Up true magnetic field in stenson remick [micro Teslas]
CONSTANT_B_FIELD_MAG = np.array([19.42900375, 1.74830615, 49.13746833])

# ============  SIM OPTIONS  ==============================================================

# total time to run sim (unrounded hours)
# HOURS = ORBITAL_PERIOD / 3600
HOURS = 10 / 3600
print("simulation time: ", HOURS, "hours")
# total time to run sim (seconds)
TF = int(HOURS * 3600)
# time step (how long between each iteration)
DT = .25
# threshold for when we consider our satellite detumbled (degrees/s)
DETUMBLE_THRESHOLD = 0.5
# convert to rad/s
if not DEGREES:
    DETUMBLE_THRESHOLD *= math.pi / 180

STATE_SPACE_DIMENSION = 7
MEASUREMENT_SPACE_DIMENSION = 6

# whether to generate new pySOL data or not
GENERATE_NEW = True
# csv to get pre-generated pysol b field from
CSV_FILE = "1_orbit_half_second" # .5 dt
# CSV_FILE = "1_orbit_tenth_second" # .1 dt

# if false, use PySOL to calculate orbital magnetic field
CONSTANT_B_FIELD = False
RW_OFF = True
SENSOR_NOISE = True
STANDSTILL = True # keep satellite in same position around the earth
# 0 = only create pdf output, 1 = show 3D animation visualization, 2 = both, 3 = none
RESULT = 0
OUTPUT_DIR = "plotOutput"
OUTPUT_FILE = "output.pdf"

# time array for our graphs (hours)
TIME_GRAPHING_ARRAY = np.arange(0, TF, DT)
TIME_GRAPHING_ARRAY = TIME_GRAPHING_ARRAY / 3600

# set gyroscope to working (true) or not working (false)
GYRO_WORKING = True

# ================  3D OPTIONS  ======================================================

# whether to run physics simulator or not
SIMULATING = True
# option to only highlight orbit path with new cams + images
cubes_path_no_cams = False
render_images = True
two_cams = True
# whether our cams should be tilted and not ram pointed
IDEAL_TILT = False
# whether to render as color cam or ir cam (hides correct group)
IR_cam = True
# how many pairs of EHS images to create (roughly, depends on timestep)
pic_count = 100
# how often to space cams along orbit
pic_interval = int(HOURS * 3600 / DT / pic_count)
if IR_cam:
    pic_width = 24
    pic_height = 32
else:
    # settings for higher quality color cam
    pic_width = 512
    pic_height = 512
# sensor width (in mm) and desired FOV
cam_FOV_vertical = 110.0
cam_FOV_horizontal = 70
sensor_width = 25.8
sensor_height = 17.8 
# angle at which our cams are mounted (degrees)
# with respect to axis want to nadir point
cam_mount_angle = 30
earth_object = "earth"
sun_earth_group = "earth_sun"
ir_earth_group = "earth_IR"

# ================  CUBESAT SYSTEM  ======================================================

# updated NearSpace intertia tensor of Cubesat body (lbs * in^2)
# CUBESAT_BODY_INERTIA = np.array([[3.02,-0.02,0.01], 
                                # [-0.02,4.13,-0.01],
                                # [0.01,-0.01,1.96]]) 
# CAD intertia (ounces * in^2)
CUBESAT_BODY_INERTIA = np.array([[250.101,4.092,0.637], 
                                [4.092,335.329,1.243],
                                [0.637,1.243,151.208]]) 
# convert to kg * m^2 from ounces * in^2
CUBESAT_BODY_INERTIA = CUBESAT_BODY_INERTIA * 1.82899783e-5

# Body inertia conversion to kg * m^2
# CUBESAT_BODY_INERTIA = CUBESAT_BODY_INERTIA * 0.00029263941
CUBESAT_BODY_INERTIA_INVERSE = np.linalg.inv(CUBESAT_BODY_INERTIA)
INCLINATION_RAD = math.radians(ORBITAL_ELEMENTS[3]) # inclination of orbit (radians)

CUBESAT_eigenvalues, CUBESAT_eigenvectors = np.linalg.eig(CUBESAT_BODY_INERTIA)

# bang-bang controller gains
# KP = .01 # for bang-bang
# KD = .009
KP = 3e4 # for normal conversion
KD = 5e4

if GYRO_WORKING:
    K = 1e-5 # old EOMS, constant B
    K = 4e-5 # for accurate mag readings
    # proposed value for k for b-cross algorithm
    # K = 2 * ((ORBITAL_ELEMENTS[1] * 1000 ) ** 3 / (GRAVITY_EARTH))**(-0.5) * (1 + math.sin(INCLINATION_RAD)) * min(CUBESAT_eigenvalues)
else:
    # K = .25e5 # proportional gain for B_dot without gyro
    K = 1.5e3 # .275e4 is best
    # for constant b-field, .2-.14e4 works
    #   it steadily decreases until 1 reaches a point it starts steadily increasing/decreasing
    # Avanzini and Giulietti (https://arc.aiaa.org/doi/10.2514/1.53074): 
    #     k =2n(1+sinζ)Imin "this is bdot"-andrew
    #     GM - grav constant, a - orbit semi-major axis, lambda - J minimum eigenvalue, n is mean motion of satellite,
    #     ζ is the inclination of the orbit with respect to the geomagnetic equator, and Imin is the value of minimum moment of inertia.
    # find minimum element of CUBE_BODY_INERTIA
    # I_min = np.min(np.diagonal(CUBESAT_BODY_INERTIA))
    # average angular velocity over one orbit
    # MEAN_MOTION = 2 * np.pi / ORBITAL_PERIOD
    # K = 2 * MEAN_MOTION * (1 + math.sin(INCLINATION_RAD)) * I_min
    # orbital angular rate (rad/s): how fast it orbits earth
    # ORBIT_RATE = math.sqrt(GRAVITY_EARTH / (ORBITAL_ELEMENTS[1] * 1000)**3)
    # K = 2 * ORBIT_RATE * (1 + math.sin(INCLINATION_RAD)) * I_min 
print("gain: ", K)

# Max torque for both Torquers
MAX_VOLTAGE = 5  # Maximum voltage [V]
# we have ~1 amp total between all torques
# MAX_CURRENT = 1 / 3  # Maximum current [A]
MAX_CURRENT = 0.4  # Maximum current [A]
RESISTANCE_MAG = 12 # Resistance [Ohm]
INDUCTANCE_MAG = 146 # Inductance [H]

#===============  AIRCORE TORQUER  =====================================================

# Magnetorquer geometry (rectangular air core)
# AIR_NUM_TURNS = 654  # total number of turns of coil
AIR_NUM_TURNS = 384  # green expiremental green air toruqer (from robby)
AIR_AREA = 0.008 # Area of magnetorquer [m^2]

# expiremental values
# want 60 mA current at max voltage
# resistance controls the max current
AIR_RESISTANCE_MAG = 80 # (Ohms)
# THIS CONTROLS RATE OF CHANGE OF CURRENT (lower = lower time constant/charging speed)
AIR_INDUCTANCE_MAG = 23 # Inductance [H]

AIR_MAX_TORQUE = 4.997917534360683e-05 # N·m
# Total Resistance: 15.692810457516336 Ohms
# Magnetic Dipole: 1.665972511453561 A·m²
# Power Dissipation at Max Usage: 2.510849673202614 W
# Power Dissipation in Watt-Hours (for 1 hour): 0.0418474945533769 Wh
# Total Wire Length: 224.18300653594764 m (edited) 

# ================  MUMETAL TORQUER  ======================================================

RELATIVE_PERM_MM = 80000  # Relative permeability of MuMetal (between 80,000 and 100,000)
FERRO_LENGTH = 7 # length of the rod [cm]
FERRO_ROD_RADIUS = 0.32 # Core rod radius [cm]
# FERRO_NUM_TURNS = 1845 # number of turns of coil
FERRO_NUM_TURNS = 2200 # expiremntal ferro torquer (from robby)
FERRO_AREA = np.pi * (FERRO_ROD_RADIUS / 100)**2 # Area of magnetorquer [m^2]

# expiremental values
# want 225 mA current at max voltage
# resistance controls the max current
FERRO_RESISTANCE_MAG = 20 # (Ohms)
# THIS CONTROLS RATE OF CHANGE OF CURRENT (lower = lower time constant/charging speed)
FERRO_INDUCTANCE_MAG = 5 # Inductance [H]

# taken from Sarah's optimizing code (with 80000 permeability)
FERRO_MAX_TORQUE = 3.185e-5 # n*m

# equations for relative permeability and demagnetizing factor from fullcalcs.py
FERRO_RADIUS = FERRO_LENGTH/FERRO_ROD_RADIUS # length-to-radius ratio of the cylindrical magnetorquer     
FERRO_DEMAG_FACTOR = (4 * np.log(FERRO_RADIUS - 1)) / (FERRO_RADIUS * FERRO_RADIUS - 4 * np.log(FERRO_RADIUS))        
# theorized epsilono should be between 100 and 300
FERRO_EPSILON = ( 1 + (RELATIVE_PERM_MM - 1) ) / (1 + FERRO_DEMAG_FACTOR * (RELATIVE_PERM_MM-1))
# alternate method by "Attitude Control by Magnetic Torquer"
# ferro_epsilon = 1 / (1/rel_perm + (((2*core_radius)**2) / (core_length**2))*(np.log(2*core_length/core_radius) - 1))

# =======  SENSORS  ==================================================

# noise sd = noise density * sqrt(sampling rate)
# vn100 imu sampling rate from user manual = 200 Hz

# mag noise density from vn100 website = 140 uGauss /sqrt(Hz)
# must convert from uGuass to microTesla
SENSOR_MAGNETOMETER_SD = (140 / 10000) * np.sqrt(200)

# gyro noise density from vn100 website = 0.0035 degree/s /sqrt(Hz)
SENSOR_GYROSCOPE_SD = 0.0035 * np.sqrt(200)
if not DEGREES:
    SENSOR_GYROSCOPE_SD = 0.0035 * np.sqrt(200) * (np.pi / 180)

EARTH_MAGNETIC_FIELD_LEO = 30e-6  # Average magnetic flux density in LEO [T]
