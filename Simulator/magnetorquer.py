'''
magnetorquer.py
Author: Daniel, Brian, Sophie

??
'''
# TODO: add docstring
from Horizon_Sensor_Sim.params import *

class Magnetorquer():
    '''Class to hold a single magnetorquer'''

    def __init__(self, n: int, area: float, k: float, epsilon: float):
        '''
        Initialize all parameters needed to model a magnetorquer
        @params:
            n (int): number of turns of coil for each magnetorquer (num of layers * num of turns per layer?)
            area (float): area of magnetorquer for each magnetorquer (m^2)
            k (float): detumbling constant gain for each magnetorquer
            epsilon (float): added multiplication term for ferromagnetic magnetorquers, 1 for air torquers
        '''
        self.n = n
        self.area = area
        self.k = k
        # accounts for the magnetic permeability and the geometry of the core. 1 for air-core magnetorquers
        self.epsilon = epsilon

        if (self.epsilon == 1) :
            self.resistance = AIR_RESISTANCE_MAG
            self.inductance = AIR_INDUCTANCE_MAG
        else:
            self.resistance = FERRO_RESISTANCE_MAG
            self.inductance = FERRO_INDUCTANCE_MAG

    def __repr__(self): # representation, returns string with details on the magnetorquer when "mag_object_name" is called
        if (self.epsilon == 1) :
            return "Air Magnetorquer: \n    Number of turns = {}\n    Area = {} m^2\n    k = {}".format(self.n, self.area, self.k)
        else: # mention special epsilon if this is a ferromagnetic magnetorquer
            return "Ferro Magnetorquer: \n    Number of turns = {}\n    Area = {} m^2\n    k = {}\n    Magnitizing factor = {}".format(self.n, self.area, self.k, self.epsilon)