'''
mag_EOMs.py
Authors: Andrew Gaylord, Michael Paulucci, Sarah Keopfer, Lauren, Kris, Daniel

Contains the satellite model for the magnetorquer-only ADCS
Magnetorquer_Sat class contains the specifications and current state for our NearSpace satellite
'''

import math
import numpy as np

import Horizon_Sensor_Sim.Simulator.magnetorquer as mag
import Horizon_Sensor_Sim.Simulator.camera as cam
from Horizon_Sensor_Sim.params import *

class Magnetorquer_Sat():
    '''
    Class that represents the specifications and current state for our NearSpace satellite
    '''
    def __init__(self, I_body:np.ndarray, magnetorquers: list[mag.Magnetorquer], w_sat: np.ndarray, B_body: np.ndarray, prevB: np.ndarray, DT: float, gyro_working: bool):
        '''
        Initialize all parameters needed to model our cubesat and its magnetorquers
        number of coils and area are arrays because our mags have different specifications

        @params:
            I_body (np.ndarray, (3x3)): moment of inertia tensor of satellite
            magnetorquers (list[mag.Magnetorquer]): list of magnetorquer objects
            w_sat (np.ndarray, (1x3)): angular velocity vector (rad/s) (radians not degrees!)
            B_body (np.ndarray, (1x3)): Magnetic field of body (Teslas)
            prevB (np.ndarray, (1x3)): previous magnetic field of body (Teslas)
            DT (float): timestep
            gyro_working (bool): whether or not the gyroscope is working
        '''
        
        self.I_body = I_body                        # Initially store moment of inertia tensor
        self.I_body_inv = np.linalg.inv(I_body)

        self.w_sat = w_sat                          # angular velocity of vector
        self.B_body = B_body

        self.mags = magnetorquers

        # # array of max torques for each magnetorquer (used to check boundaries)
        # self.max_torque = np.array([
        #     AIR_MAX_TORQUE if mag.epsilon == 1 else FERRO_MAX_TORQUE
        #     for mag in self.mags
        # ])

        # array of resistances and inductances for each magnetorquer
        self.resistances = np.array([mag.resistance for mag in self.mags])
        self.inductances = np.array([mag.inductance for mag in self.mags])

        # dt and prev B field for calculations with no gyroscope
        self.dt = DT
        self.prevB = prevB
        self.gyro_working = gyro_working

        # proportional and derivative for bang bang/pd controller
        self.kp = KP
        self.kd = KD

        # what state our satellite is in ("detumble", "search", "point")
        self.state = STARTING_PROTOCOL

        # create two camera objects (facing either direction)
        self.cam1 = cam.Camera()
        self.cam2 = cam.Camera()

        # all_mags = [0 for _ in range(3)] # create an array holding 3 Magnetorquer objects

        # for index, magnetorquer in enumerate(all_mags):
        #     # update the properties of each magnetorquer
        #     magnetorquer = mag.Magnetorquer(n = n[index], area = area[index], k = k[index], B_body = B_body[index], w_sat = w_sat[index], epsilon = 1)
        #     if (index != 2): # third magnetorquer is the non-ferromagnetic one, calculate special epsilon for the others
        #         ratio = mag_length/core_radius # length-to-radius ratio of the cylindrical magnetorquer
        #         
        #         #demag_factor = (4*math.log(ratio - 1)) / ((ratio*ratio) - (4*math.log(ratio)))
        #         demag_factor = calculate_demagnetizing_factor(ratio) # pull equation demagnetizing facotr from fullcalcs.py
        #         
        #         ferro_epsilon = 1 + ((rel_perm - 1)/(1 + demag_factor*(rel_perm-1)))
        #         magnetorquer.epsilon = ferro_epsilon
    
    def momentToVoltage(self, moment: np.ndarray) -> np.ndarray:
        '''
        Converts a desired magnetic moment to the voltage needed to generate that moment

        @params:
            moment (np.ndarray, (1x3)): desired magnetic moment (Amps * m^2)
        @returns:
            voltage (np.ndarray, (1x3)): voltage needed to generate that moment (Volts)
        '''
        
        # calculate the voltage needed to generate the desired magnetic moment
        current = np.zeros((3))
        # find current for 2 ferro and 1 air core magnetorquers using dipole / nA*epsilon
        for i in range(3):
            current[i] = moment[i] / (self.mags[i].n * self.mags[i].area * self.mags[i].epsilon)

        # convert current to voltage using Ohm's Law
        voltage = np.array(current) * self.resistances

        return np.array(voltage)
