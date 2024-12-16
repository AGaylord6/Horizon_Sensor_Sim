# Horizon_Sensor_Sim
VFX/IrishSat project to generate artificial Earth Horizon Sensor (EHS) images using Python and Maya. 

Interface_script.py uses IrishSat's orbital simulation library (PySOL) to generate a specified flight path, then it creates cameras along the route and renders a realistic (hopefully) image of what the satellite would be seeing at the point in time. It finds location and velocity data so that it can align the front of the camera with the direction that our satellite is moving (called the RAM vector). 

The script can simulate normal cameras or infrared (IR) cameras (which have worse resolution). It can also simply create cubes along an path to help visualize the so-called "orbital elements" that define an orbit. 

IrishSat simulated orbital library: https://github.com/ND-IrishSat/PySOL
