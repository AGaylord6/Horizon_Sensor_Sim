'''
BangBang.py
Authors: Peyton Reynolds, Lauren Catalano, David Scully, Luke Tocco, Michael Kuczun

Bang Bang algorithm works to minimize error quaternion by writing current to magnetorquers to actuate the satellite

'''

import numpy as np
import math
import sys, os

# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Horizon_Sensor_Sim.params import *
from Horizon_Sensor_Sim.Simulator.all_EOMs import normalize, delta_q, quaternionMultiply


def BangBang (current, target, mag_sat):
    '''
    Bang-Bang/proportional-derivative controller to orient us towards a target quaternion
    
    @params:
        current (np.ndarray): current quaternion of satellite (1x4)
        target (np.ndarray): target quaternion of satellite (1x4)
        mag_sat (Magnetorquer_Sat): object from sat_model.py that represents our satellite and its magnetorquers, including gains and mag specs
    @returns:
        voltage_in (np.ndarray): vector representing voltage we are sending along all three magnetorquers (Volts)
    '''
    
    # Find the error quaternion between current and target quaternion
    # represents the difference in orientation; [1, 0, 0, 0] meaning that they're aligned
    error_quat = delta_q(current, target)
    
    # Define the torque using the error quaternion and angular velocity (equation 7.7 from Fund of Spacecraft Att Det)
    #   Derivative term  responds to how fast the error quaternion is changing over time (which is related to how fast we're spinning)
    #   this allows us to anticipate and dampen rapid changes, opposing quick changes and preventing overshooting
    torque = - mag_sat.kp * error_quat[1:4] - mag_sat.kd * mag_sat.w_sat

    # find part of torque that is perpendicular to B
    # B_norm_sq = np.dot(B, B)
    # if B_norm_sq == 0:
    #     raise ValueError("Magnetic field vector cannot be zero.")
    
    # # Project desired torque onto the plane perpendicular to B
    # T_d_perp = T_d - (np.dot(T_d, B) / B_norm_sq) * B
    
    # # Compute the required magnetic moment
    # m = np.cross(B, T_d_perp) / B_norm_sq

    # Define the magnetic moment by taking a cross product of the magnetic field with the previously defined torque
    # TODO: only works if they're all orthogonal??
    # by doing this, our actual torque will not be aligned with desired torque if they're not orthogonal
    # https://math.stackexchange.com/questions/32600/whats-the-opposite-of-a-cross-product
    m = np.cross( mag_sat.B_body, torque )

    # normalize the magnetic moment so that we're just getting a direction without magnitude
    # m_unit = normalize(m)

    # convert magnetic moment to voltage
    voltage = mag_sat.momentToVoltage(m)

    # OR, just scale our max voltage according to the magnitude of the magnetic moment
    # voltage = MAX_VOLTAGE * m_unit

    # OR, compute the scaling factor to make the largest component equal to MAX_VOLTAGE
    # largest_component = np.max(np.abs(m))
    # scaling_factor = MAX_VOLTAGE / largest_component
    
    # voltage = m * scaling_factor
    

    return voltage













'''
def BangBang( kp, kd, err_quat, omega, b_field ):

    # Define the torque using the error quaternion and angular velocity (equation 7.7 from Fund of Spacecraft Att Det)
    L = - kp * err_quat[1:4] - kd * omega

    # Define the magnetic moment by taking a cross product of the magnetic field with the previously defined torque
    m = np.cross( b_field, L )

    # Take the unit vector of the magnetic moment so that we're just getting a direction without magnitude
    m_magnitude_squared = pow( m[0], 2 ) + pow( m[1], 2 ) + pow( m[2], 2 )
    m_magnitude = math.sqrt( m_magnitude_squared )
    m_unit = m / m_magnitude

    # Initialize the currents for the magnetorquers
    I = np.zeros(3)

    # Maximize the current on the highest magnetic moment direction to set the needed magnetic moment
    # Scale the rest of the currents according to their portion of the total unit vector
    if m_unit[0] > m_unit[1] and m_unit[0] > m_unit[2]:
        I[0] = 0.4
        I[1] = 0.4 * (m_unit[1]/m_unit[0])
        I[2] = 0.4 * (m_unit[2]/m_unit[0])
    elif m_unit[1] > m_unit[2]:
        I[1] = 0.4
        I[0] = 0.4 * (m_unit[0]/m_unit[1])
        I[2] = 0.4 * (m_unit[2]/m_unit[1])
    else:
        I[2] = 0.4
        I[0] = 0.4 * (m_unit[0]/m_unit[2])
        I[1] = 0.4 * (m_unit[1]/m_unit[2])



    return I

'''
