'''
B_dot.py
Author: Peyton Reynolds, Michael Paulucci, Andrew Gaylord

Contains the B dot algorithm used for detumbling our satellite with magnetorquers
References the Magnetorquer_Sat class in mag_EOMs.py

'''

from sat_model import Magnetorquer_Sat
from all_EOMs import normalize  # is this needed
import numpy as np

def B_dot(sat: Magnetorquer_Sat):
    '''
    B dot algorithm for detumbling our satellite

    @params:
        sat: object that represents our satellite and its magnetorquers. Includes:
            w_sat: angular velocity vector from sensor (rad/s)
            B_body: Magnetic field of body from sensor (Teslas)
            gyro_working: whether gyroscope is working or not (bool)
            prevB: previous magnetic field of body (Teslas)
            dt: timestep (UNIT?)
            3 magnetorquer objects, each with:
                n_sat: Number of coils for magnetorquer (int)
                A_sat: Area of the three magnetorquers resepctively (m^2)
                k: proportional factor/gain constant

    @returns:
        voltage_in: vector representing voltage we are sending along all three magnetorquers (Volts)
    '''

    # find desired magentic dipole/moment needed to counteract angular velocity
    B_magnitude_squared = np.linalg.norm(sat.B_body)**2
    if B_magnitude_squared == 0:
        print("Magnetic field vector magnitude cannot be zero.")
    
    # use b-cross equation when we have access to magnetic field data
    if sat.gyro_working:
        # compute magnetic moment using the b-cross control law: - (k / ||B||^2) * (B x w)
        # units: Amps * m^2 (B_body must be in teslas)
        b_dot_term = np.cross(sat.B_body, sat.w_sat)
        desiredMagneticMoment = np.array([-(sat.mags[i].k / B_magnitude_squared) * b_dot_term[i] for i in range(len(b_dot_term))])
    else:
        # if gyroscope is off, use actual derivative of B field
        # Computer magnetic moment using the control law without w: - k * B'
        # TODO: make sure prevB is being updated correctly
        # TODO: research methods to smooth noise out: Savitzky-Golay Filter, low pass filter, kalman, etc
        dB = ( sat.B_body - sat.prevB ) / sat.dt
        desiredMagneticMoment = - sat.mags[0].k * dB

    # print("magnetic moment: ", magneticMoment)
    
    current_in = np.zeros((3))
    # find current for 2 ferro and 1 air core magnetorquers using dipole / nA*epsilon
    for i in range(3):
        current_in[i] = desiredMagneticMoment[i] / (sat.mags[i].n * sat.mags[i].area * sat.mags[i].epsilon)

    # convert to voltage (12 is system's resistance)
    # note: alternatively, don't convert to voltage and tweak k accordingly to account for it
    voltage_in = np.array(current_in) * 12

    return np.array(voltage_in)