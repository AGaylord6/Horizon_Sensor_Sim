'''
image_processing.py
Authors: Andrew, Brian, Kris, Rawan, Daniel, Chau

Image processing script for finding horizon edges, regression line, pitch and roll of satellite
Input: sample satellite Earth Horizon Sensor (EHS) images

https://learnopencv.com/edge-detection-using-opencv/
https://docs.opencv.org/4.0.0/d7/de1/tutorial_js_canny.html
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10375389

'''

import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import math
from Horizon_Sensor_Sim.params import *

# HEIGHT = 450 * 1000 # meters
IMAGE_WIDTH = pic_width
IMAGE_HEIGHT = pic_height
FOV = cam_FOV_vertical # 110 # degrees
SENSOR_WIDTH = sensor_width # mm # used to be 34
FOV_rad = math.radians(FOV) # radians
# focal length = distance from cam lens to image sensor
# should this be aperature or sensor width?
# FOCAL_LENGTH = SENSOR_WIDTH / (2 * math.tan(FOV_rad / 2)) # 11.9 (mm?)
FOCAL_LENGTH = 5.8 # mm
# aperture (f-stop) = focal length / entrace pupil diameter
# resolution of image in degrees per pixel
PIXEL_HEIGHT = FOV / IMAGE_HEIGHT

def processImage(image=None, degree=1):
    '''
    Given a Earth Horizon Sensor (EHS) image, find the line that best fits the horizon
    and return the pitch, roll, and alpha of the satellite

    @params:
        image (24x32 pixels): numpy array of EHS pixels
        degree (optional): degree of polynomial to fit
    @returns:
        roll (float): rotation about x axis. How tilted each line is. 0 = flat horizon. (degrees)
        pitch (float): angle of pointing up and down based on center of image. 0 = horizon is centered. (degrees)
        alpha (float): the percentage (as a decimal) of the image filled by the Earth
        edges (1x4 array): top, right, bottom, left edges respectively. All values between [0-1] representing how much earth is on that edge
    '''

    # ============ IMAGE GENERATION ================================

    if type(image) == type(None):
        image_directory = "images"
        # image_name = "ehs_ir_tilted_4_cam2.png" # infrared photo
        # image_name = "ehs_ir_tilted_20.png" # infrared photo tilted
        image_name = "ehs21_IR_second_1.png" # first cam photo
        # image_name = "ehs_ir_earth.png" # infrared photo of only earth
    
        # Get the absolute path to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the image
        image_path = os.path.join(script_dir, image_directory, image_name)
    
        # read our image and convert to to grayscale
        img = cv2.imread(image_path)

        print(f'{image_path} loaded by OpenCV successfully!')
    else:
        img = image
        
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # simulate our sensor output better by unfocusing and adding noise to our image
    smoothed_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # add salt-pepper noise to image (try to mimic image in nearspace controls doc)
    # adding uniform noise to the pictures 24 by 32
    uni_noise = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH),dtype=np.uint8)
    # create uniform noise (every number has equal chance)
    cv2.randu(uni_noise,0, 255) # low, high
    uni_noise_factor = 0.01
    uni_noise=(uni_noise * uni_noise_factor).astype(np.uint8)
    # create gaussian (normal) noise
    normal_noise = np.zeros_like(gray_img,dtype=np.uint8)
    normal_noise_factor = 3
    cv2.randn(normal_noise, 0, normal_noise_factor) # mean, stddev
    # add uniform + normal noise for better randomness
    noisy_img = np.clip(cv2.add(smoothed_img, uni_noise), 0, 255)
    noisy_img = np.clip(cv2.add(noisy_img, normal_noise), 0, 255)

    if img is None:
        print("Image not loaded correctly!")
        return -1, -1, 0.0, [-1, -1, -1, -1]
    
    # ============= IMAGE PROCESSING =================================

    # Apply Gaussian blur filter to reduce noise
    smoothed_img = cv2.GaussianBlur(noisy_img, (3, 3), 0)

    # Adjust the contrast and brightness
    alpha = 6  # Contrast control (1.0 is no change)
    beta = 0    # Brightness control (0 is no change)
    contrasted_img = cv2.convertScaleAbs(smoothed_img, alpha=alpha, beta=beta)

    # Apply Sobel edge detector: find intensity gradient for each pixel
    #     Look for large changes in pixel intensity in x and y direction
    # sobelx = cv2.Sobel(src=gray_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel on the X axis (essentially the X partial derivative)
    # sobely = cv2.Sobel(src=gray_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) 
    # Combined X and Y Sobel Edge Detection (fxy partial derivative)
    # TODO: try different values, smoothed_img. Maybe print through cv2.imshow()?
    sobelxy = cv2.Sobel(src=smoothed_img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    # Apply Canny edge detector: run sobel, suppress false edges, apply hysteresis thresholding
    #     If gradient is above threshold, those pixels are included in edge map
    #     If gradient is below, the pixels are suppressed and excluded from the final edge map
    #     Between thresholds = ‘weak’ edges, hysteresis mechanism will detect which belong in edge map
    # TODO: see if l2gradient parameter is needed
    canny_edges = cv2.Canny(smoothed_img, threshold1=50, threshold2=150)

    # Create another canny edge image using the increased contrast image "adjusted_img"
    canny_edges2 = cv2.Canny(contrasted_img, threshold1=50, threshold2=150)

    # TODO: impliment custom edge detection method (column based)
    custom_edges = np.array(smoothed_img)
    # transcribe the image to an array
    # for row in custom_edges:
        # lowThreshold = 0
        # highThreshold = 80
        # for pixel in row:
            # pixel_value = smoothed_img[row, pixel]
            # print(pixel_value)
            # this prints like a million zeroes so uh dont uncomment it yet
            # if(pixel_value >= lowThreshold):
            #    if(pixel_value <= highThreshold):
            ###  print(pixel_value)
            #    map(column, row as an edge pixel)

            # two scenarios of the if statements: either the value
            # of pixel may be higher than the threshold value or
            # may be less than the threshold value

    # TODO: mask edge pixels around the point closest to the center of the image??

    # extract a list of edge pixels from canny method
    edge_pixels = np.where(canny_edges != 0)
    edge_coordinates = np.array(list(zip(edge_pixels[1], edge_pixels[0])))
    # print("edge pixel coordinates: ", coordinates)
    # print("number of edge pixels: ", len(coordinates))

    if len(edge_coordinates) == 0:
        # print("No edge pixels found!")
        return -1, -1, 0, [-1, -1, -1, -1]

    x = edge_coordinates[:, 0]
    y = edge_coordinates[:, 1]

    # find the average brightness of horizon pixels to use as threshold to differentiate between space and earth
    #   this will allow us another method to recognize horizon vs space pixels
    total_edge_brightness = 0.0
    for i in range(len(y)):
        total_edge_brightness += smoothed_img[y[i]][x[i]] # WARNING: watch out for 8bit int overflow

    average_brightness = total_edge_brightness / float(len(y))
    threshold_brightness = average_brightness # add offset here if neccessary
    # print("Average intensity of edge pixels: ", average_brightness)

    #Finds the brightness of each edge, to be used to find the orientation of the sattelite such that earth is on the bottom half of the photo
    #top edge
    total_top_brightness = 0
    for column in range(len(smoothed_img[0])):
        total_top_brightness += float(smoothed_img[0][column])
    top_average_brightness = total_top_brightness/IMAGE_WIDTH
    # print("Top edge brightness is ", top_average_brightness)

    #bottom edge
    total_bottom_brightness = 0
    for column in range(len(smoothed_img[IMAGE_HEIGHT-1])):
        total_bottom_brightness += float(smoothed_img[IMAGE_HEIGHT-1][column])
    bottom_average_brightness = total_bottom_brightness/IMAGE_WIDTH
    # print("Bottom edge brightness is ", bottom_average_brightness)

    #left edge
    total_left_brightness = 0
    for row in range(len(smoothed_img)):
        total_left_brightness += float(smoothed_img[row][0])
    left_average_brightness = total_left_brightness/IMAGE_HEIGHT
    # print("Left edge brightness is ", left_average_brightness)

    #right edge
    total_right_brightness = 0
    for row in range(len(smoothed_img)):
        total_right_brightness += float(smoothed_img[row][IMAGE_WIDTH-1])
    right_average_brightness = total_right_brightness/IMAGE_HEIGHT
    # print("Right edge brightness is ", right_average_brightness)

    edges = [top_average_brightness, right_average_brightness, bottom_average_brightness, left_average_brightness]
    edges = edges / np.linalg.norm(edges)

    # find the percentage of the image that is the Earth (alpha)
    alpha = 0
    total_pixels = IMAGE_WIDTH * IMAGE_HEIGHT
    # values_above_threshold = []
    num_pixels_above_threshold = 0
    for w in range(IMAGE_WIDTH):
        for h in range(IMAGE_HEIGHT):
            # for every pixel, check if it is above the threshold
            if smoothed_img[h][w] > threshold_brightness:
                # values_above_threshold.append(smoothed_img[h][w])
                num_pixels_above_threshold += 1

    alpha = float(num_pixels_above_threshold) / float(total_pixels)
    # print("Alpha: ", alpha)

    # use numpy polynomial solver to find regression line
    coef = np.polyfit(x, y, degree)
    a = coef[0] #this is m
    b = coef[1] #this is c

    # find intersection point from center of image to horizon line
    # x_p = - (a * b) / (1 + a**2)
    # y_p = - b / (1 + a**2)
    # roll = math.degrees(math.atan(abs(x_p) / abs(y_p))) # according to article
    # roll = math.degrees(math.atan2(x_p, y_p))

    # tilt of horizon: only depends on slope. TODO: define bounds
    # positive regression line slope means positive roll
    # roll = math.degrees(-math.atan(a))
    roll = math.degrees(math.atan2(-a, 1))  # atan2(y, x) is more stable
    # print("ROLL: ", roll)

    x_c = IMAGE_WIDTH / 2
    y_c = IMAGE_HEIGHT / 2
    # find shortest perpendicular distance between center of image and horizon line (pixels)
    # p = - (a * x_c - y_c + b) / math.sqrt(a**2 + 1) # gives us positives/negatives
    # print("Pixels from center to closest point on horizon line: ", p)

    # distance to horizon from nadir
    # c = math.sqrt((EARTH_RADIUS + HEIGHT)**2 - EARTH_RADIUS**2)
    # horizon is not on same level as nadir, as surface drops away as you look further
    # this offset is how many degrees below the center of the image the horizon will naturally appear if we're pointed straight down
    # O = math.degrees(np.arctan2(c, EARTH_RADIUS))
    # print("Offset angle: ", O)
    # angle from center of image to closest point on horizon line. NOTE: altitude + mount tilt can be included to convert from camera's frame
    # pitch = p * PIXEL_HEIGHT # - O
    # print("PITCH: ", pitch)
    # this is equivalent to multiplying by pixel height, but uses trig instead
    # pitch = math.degrees(math.atan(p / FOCAL_LENGTH)) # - O

    # Compute the midpoint of the detected horizon line
    line_midpoint_x = np.mean(x)
    line_midpoint_y = np.mean(y)

    # Calculate yaw: Ratio of x-direction offset from center to total width, converted to degrees
    # Positive = right
    # yaw_ratio = (line_midpoint_x - x_c) / IMAGE_WIDTH
    # yaw = yaw_ratio * FOV_x
    
    # Calculate pitch: Ratio of y-direction offset from center to total height, converted to degrees
    pitch_ratio = (y_c - line_midpoint_y) / pic_height
    pitch = pitch_ratio * cam_FOV_vertical

    return roll, pitch, alpha, edges


if __name__ == "__main__":
    processImage()