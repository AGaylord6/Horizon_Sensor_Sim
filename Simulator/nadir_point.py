'''
nadir_point.py
Authors: Andrew, Payton, David, Sean, Lauren, Luke

Protocol for using our two Earth Horizon Sensors (EHS) to determine the direciton of the earth (nadir)
Calls image processing to find orientation and PID to move us towards desired oreintation (tries to equalize both cams)

'''

import math
import numpy as np
# import quaternion
from Horizon_Sensor_Sim.Simulator.image_processing import *
from Horizon_Sensor_Sim.Simulator.BangBang import BangBang
from Horizon_Sensor_Sim.Simulator.all_EOMs import normalize


def euler_to_quat(roll, pitch, yaw):
    '''
    Convert Euler angles (roll, pitch, yaw) to a quaternion in (w, x, y, z) order.

    @params:
        roll: The roll (rotation around x-axis) angle in radians.
        pitch: The pitch (rotation around y-axis) angle in radians.
        yaw: The yaw (rotation around z-axis) angle in radians.

    @returns
        qw, qx, qy, qz: The orientation in quaternion [w,x,y,z] format
    '''
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    
    return np.array([qw, qx, qy, qz])


def nadir_point(mag_sat):
    '''
    Given the current state of the satellite (with image processing already stored in cam1 and cam2 objects),
    try to center our cams and achieve nadir (earth) pointing
    Note: quaternions are based on the center of the image (which is [1, 0, 0, 0])

    @params:
        mag_sat (Magnetorquer_satellite object): encapsulates current state of our cubesat
    '''

    # TODO: handle when 1 cam is upside down (want to rotate through earth)
    #   they can be facing different directions because the FOV's overlap

    # if upside down, rotate hard through Y (or trigged axis of fullest edge -- or roll!) of one pointing more towards earth?
    #   or just base on one that has earth in bottom of horizon

    # observation: one can be near nadir while other sees entirely earth (somehow)
    
    # create quaternion from first EHS
    roll1 = math.radians(mag_sat.cam1.roll)
    pitch1 = math.radians(mag_sat.cam1.pitch)
    q1 = normalize(euler_to_quat(roll1, pitch1, 0.0))

    # create quaternion from second EHS
    roll2 = math.radians(mag_sat.cam2.roll)
    pitch2 = math.radians(mag_sat.cam2.pitch)
    q2 = normalize(euler_to_quat(roll2, pitch2, 0.0))

    # edges are top, right, bottom, left intensities (0-1)
    if mag_sat.cam1.edges[0] > mag_sat.cam1.edges[2] and mag_sat.cam2.edges[0] < mag_sat.cam2.edges[2]:
        # if first cam is upside down (bottom less than top) and second if not
        current_orientation = q2
    else:
        current_orientation = q1

    # define target orientation as ~24 degrees pitched up (everything else = 0)
    # if current quat is set to [1, 0, 0, 0], this incites a constant angular y velocity
    # ALPHA$ method (alpha% = 70.2%)
    target_orientation = normalize(euler_to_quat(0.0, math.radians(24), 0.0))

    # try to even the two cams if near nadir pointing
    # target_orientation = normalize(euler_to_quat((roll1+roll2)/2, (pitch1+pitch2)/2, 0.0))
    # "slerp" is a method of getting midpoint of quaternions but couldn't get working
    # target_quaternion = np.quaternion.slerp_evaluate(q1, q2, 0.5)

    # print("target: ", target_orientation)
    # print("current: ", current_orientation)

    # get voltages required to move us towards target quaternion
    # current_orientation = np.array([1.0, 0.0,0.0,0.0])
    voltage = BangBang(current_orientation, target_orientation, mag_sat)

    return voltage




# def axisangle_to_q(v, theta):
#     v = normalize(v)
#     x, y, z = v
#     theta /= 2
#     w = math.cos(theta)
#     x = x * math.sin(theta)
#     y = y * math.sin(theta)
#     z = z * math.sin(theta)
#     return w, x, y, z

# def q_conjugate(q):
#     w, x, y, z = q
#     return (w, -x, -y, -z)

# def qv_mult(q1, v1):
#     q2 = np.concatenate(([0.0], v1))
#     return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

# def q_mult(q1, q2):
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#     return w, x, y, z


if __name__ == '__main__':
    nadir_point()
