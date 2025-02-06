'''
camera.py
Author: Andrew Gaylord

Represents our Earth Horizon Sensor (EHS) cams and their current image processing output

'''

class Camera():

    def __init__(self):

        # store pitch and roll of found horizon (None = all earth or all space)
        self.pitch = -1
        self.roll = -1
        # store the percentage of frame filled with earth
        self.alpha = -1
        # array of the average intensity of our 4 edges
        self.edges = []