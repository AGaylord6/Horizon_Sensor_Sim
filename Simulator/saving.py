'''
saving.py
Author: Andrew Gaylord

contains the saving functionality for kalman filter visualization
saves graphs to png and then embeds them in a pdf with contextual information

'''

import os
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import os
import sys

# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Horizon_Sensor_Sim.params import OUTPUT_DIR, ORBITAL_PERIOD, ORBITAL_ELEMENTS, DEGREES

def saveFig(fig, fileName):
    '''
    saves fig to a png file in the outputDir directory with the name fileName
    also closes fig
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    saveDirectory = os.path.join(my_path, OUTPUT_DIR)

    fig.savefig(os.path.join(saveDirectory, fileName), bbox_inches='tight')

    plt.close(fig)


def savePDF(outputFile, pngDir, sim):
    '''
    creates a simulation report using FPDF with all PNGs found in pngDir
    Describes the different graphs and their significance

    @params:
        outputFile: name of pdf to be generated
        pngDir: name of folder where graph PNGs are found
        sim: Simulator object with sim info
        controller: PIDController object with weights info. If = None, controls info is not printed
        target: our target quaternion for this simulation
        sum: statistical tests sum
        printTests: whether statistical tests were run or not (true = print test info)
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    pngDirectory = os.path.join(my_path, pngDir)

    # create the PDF object
    pdf = FPDF()
    title = "Detumbling Simulation Report"
    pdf.set_author("Andrew Gaylord")
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    x_offset = 12
    y_pic_offset = 134

    # title and document details
    pdfHeader(pdf, title)

    pdf.image(os.path.join(pngDirectory, "magData.png"), x=x_offset, y=pdf.get_y(), w=180)
    pdf.ln(y_pic_offset)
    pdf.image(os.path.join(pngDirectory, "B_earth.png"), x=x_offset, y=pdf.get_y(), w=180)
    # pdf.image(os.path.join(pngDirectory, "gyroData.png"), x=x_offset, y=pdf.get_y(), w=180)

    pdf.add_page()


    pdfHeader(pdf, "Orientation and Angular Velocity")

    pdf.image(os.path.join(pngDirectory, "Quaternion.png"), x=x_offset, y=pdf.get_y(), w=180)
    pdf.ln(y_pic_offset)
    pdf.image(os.path.join(pngDirectory, "Velocity.png"), x=x_offset, y=pdf.get_y(), w=180)

    pdf.add_page()

    pdfHeader(pdf, "Magnetorquer Information")

    # magText = f"""Magnetorquer results with gain k = {sim.mag_sat.k}"""
    # TODO: print more info (and magnitude of current, velocity)
    # pdf.multi_cell(0, 5, magText, 0, 'L')

    pdf.image(os.path.join(pngDirectory, "Total_Power_Output.png"), x = x_offset, y = pdf.get_y(), w = 180)

    pdf.ln(y_pic_offset)
    pdf.image(os.path.join(pngDirectory, "Power_Output.png"), x = x_offset, y = pdf.get_y(), w = 180)

    pdf.add_page()
    
    pdf.image(os.path.join(pngDirectory, "Currents.png"), x = x_offset, y = pdf.get_y(), w = 180)

    pdf.ln(y_pic_offset)

    pdf.image(os.path.join(pngDirectory, "Torques.png"), x = x_offset, y = pdf.get_y(), w = 180)

    pdf.add_page()
    
    pdf.image(os.path.join(pngDirectory, "Voltages.png"), x = x_offset, y = pdf.get_y(), w = 180)

    pdf.ln(y_pic_offset)

    # eulerText = f"""Our filtered orientation represented by Euler Angles (counterclockwise rotation about x, y, z). Can bug out sometimes. Near 180 degrees (pi) is the same as zero. """
    # pdf.multi_cell(0, 5, eulerText, 0, 'L')
    # pdf.image(os.path.join(pngDirectory, "Euler.png"), x=x_offset, y=pdf.get_y(), w=180)
    pdf.image(os.path.join(pngDirectory, "Pitch_Roll.png"), x = x_offset, y = pdf.get_y(), w = 180)
    
    pdf.add_page()

    pdf.image(os.path.join(pngDirectory, "Edges1.png"), x = x_offset, y = pdf.get_y(), w = 180)

    pdf.ln(y_pic_offset)
    
    pdf.image(os.path.join(pngDirectory, "Edges2.png"), x = x_offset, y = pdf.get_y(), w = 180)
    
    pdf.add_page()

    pdfHeader(pdf, "General Info")

    # set numpy printing option so that 0's don't have scientific notation
    np.set_printoptions(formatter={'all': lambda x: '{:<11d}'.format(int(x)) if x == 0 else "{:+.2e}".format(x)})
    pdf.set_font("Arial", size=13)

    starting_speed = sim.states[0][4:]
    if not DEGREES: 
        starting_speed *= 180/np.pi

    infoText = f"""Starting speed: {starting_speed} degrees/s.

Total simulation time: {float(sim.n) * sim.dt / 3600} hours

Orbits completed during simulation: {round((sim.n * sim.dt)/ORBITAL_PERIOD, 4)} orbits.

Hours to detumble: {round(sim.finishedTime/3600, 4)} hours.

Orbits to detumble: {round(sim.finishedTime/ORBITAL_PERIOD, 4)} orbits.

Power consumed to detumble (Total Energy): {int(sim.energy)} Jules

Orbital elments: {ORBITAL_ELEMENTS}
    These define our simulated orbit (see sol_sim.py in PySOL for more info)

B-dot proportional gain: k = {sim.mag_sat.mags[0].k}

Bang-Bang proportional gain: kp = {sim.mag_sat.kp}
Bang-Bang derivative gain: kd = {sim.mag_sat.kd}

Satellite info:

{sim.mag_sat.mags[0]}

{sim.mag_sat.mags[1]}

{sim.mag_sat.mags[2]}

"""

    pdf.multi_cell(0, 5, infoText, 0, 'L')

    # output the pdf to the outputFile
    outputPath = os.path.join(my_path, outputFile)
    pdf.output(outputPath)


def pdfHeader(pdf, title):
    '''
    insert a header in the pdf with title
    '''

    pdf.set_font('Arial', 'B', 14)
    # Calculate width of title and position
    w = pdf.get_string_width(title) + 6
    pdf.set_x((210 - w) / 2)
    # Colors of frame, background and text
    pdf.set_draw_color(255, 255, 255)
    pdf.set_fill_color(255, 255, 255)
    # pdf.set_text_color(220, 50, 50)
    # Thickness of frame (1 mm)
    # pdf.set_line_width(1)
    pdf.cell(w, 9, title, 1, 1, 'C', 1)
    # Line break
    pdf.ln(2)

    # return to normal font
    pdf.set_font("Arial", size=11)


def savePNGs(outputDir):
    '''
    saves all currently open plots as PNGs in outputDir and closes them
    '''

    # absolute path to current directory
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    saveDirectory = os.path.join(my_path, outputDir)
    
    # get list of all figures
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    numPlots = 0
    # iterate over and save all plots tp saveDirectory
    for fig in figs:  
        numPlots += 1
        
        # save and close the current figure
        fig.savefig(os.path.join(saveDirectory, "plot" + str(numPlots) + ".png"))
        # fig.savefig(saveDirectory + "plot" + str(numPlots) + ".png")

        plt.close(fig)


def openFile(outputFile):
    # open the pdf file
    subprocess.Popen([outputFile], shell=True)


def clearDir(outputDir):

    # create the output directory if it doesn't exist
    my_path = os.path.dirname(os.path.abspath(__file__)) 
    saveDirectory = os.path.join(my_path, outputDir)
    if not os.path.exists(saveDirectory):
        os.makedirs(saveDirectory)

    # removes all files in the output directory
    files = os.listdir(saveDirectory)
    for file in files:
        file_path = os.path.join(saveDirectory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    

