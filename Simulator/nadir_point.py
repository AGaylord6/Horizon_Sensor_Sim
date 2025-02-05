'''
nadir_point.py
Authors: Andrew, Payton, David, Sean, Lauren, Luke

Protocol for using our two Earth Horizon Sensors (EHS) to determine the direciton of the earth (nadir)
Calls image processing to find orientation and PID to move us towards desired oreintation (tries to equalize both cams)

'''

import math
import numpy as np
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

    @params:
        mag_sat (Magnetorquer_satellite object): encapsulates current state of our cubesat
    '''
    
    roll1 = math.radians(mag_sat.cam1.roll)
    pitch1 = math.radians(mag_sat.cam1.pitch)

    # Current orientation quaternion
    current_orientation = normalize(euler_to_quat(roll1, pitch1, 0.0))

    # define target orientation as 20 degrees pitched up (everything else = 0)
    target_orientation = normalize(euler_to_quat(0.0, 0.436, 0.0))
    # print("target: ", target_orientation)
    # print("current: ", current_orientation)

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
