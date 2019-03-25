# ---------------------------------------------------------- #
# This is a code to perform spectrophotometric calibration.  #
# It was designed to calibrate spectral data from the Anglo  #
# Australian Telescope by matching it to near simultaneous   #
# photometric observations using DECam on the Blanco         #
# Telescope as part of the OzDES Reverberation Mapping       #
# Program.  The bulk of the calculations are defined in the  #
# file calibSpec_calc.py.  This code defines file locations, #
# reads in the data, and calls the calibration function.     #
# Unless otherwise noted this code was written by            #
# Janie Hoormann.                                            #
# ---------------------------------------------------------- #

import numpy as np
import calibSpec_calc as calc
import matplotlib.pyplot as plt


# First define where all of the data can/will be found

# Define where the transmission function is stored
bands = [b'g', b'r', b'i']
filters = {b'g': '../OzDES_Pipeline/RMPipeline/input/DES_g_y3a1.dat',
           b'r': '../OzDES_Pipeline/RMPipeline/input/DES_r_y3a1.dat',
           b'i': '../OzDES_Pipeline/RMPipeline/input/DES_i_y3a1.dat'}
centers = [4730, 6420, 7840]

# Define where spectra are stored and file name format: name = spectraBase + ID + spectraEnd
spectraBase = "../OzDES_Pipeline/RMPipeline/input/SVA1_COADD-"
spectraEnd = ".fits"

# Define where photometry are stored and file name format
photoBase = "../OzDES_Pipeline/RMPipeline/input/"
photoEnd = "_lc.dat"

# Define the name of the file that holds the list of sources to calibrate, which we want to be sure is an array
idNames = "../OzDES_Pipeline/RMPipeline/input/testNames.txt"
names = np.genfromtxt(idNames)

if names.size == 1:
    names = np.array([names])

# Define the name of the place you want the output data stored
outDir = "../"

# Do you want calibration plots - if so set the flag to the place where they should be saved, otherwise set it to false
# plotFlag = False
plotFlag = "../"

# Now we actually call functions and do calculations
for i in range(1):
    obj_name = str(int(names[i]))

    spectraName = spectraBase + obj_name + spectraEnd
    photoName = photoBase + obj_name + photoEnd

    print("Input Spectra Name: %s" % spectraName)
    spectra = calc.Spectrumv18(spectraName)
    plt.plot(spectra.wavelength, spectra.wavlength)

    calc.mark_as_bad(spectra.flux, spectra.variance)

    print("Input Photometry Name: %s" % photoName)
    photo = np.loadtxt(photoName, dtype={'names':('Date', 'Mag', 'Mag_err', 'Band'),
                                         'formats':(np.float, np.float, np.float, '|S15')}, skiprows=1)

    calc.calibSpec(obj_name, spectra, photo, spectraName, photoName, outDir, bands, filters, centers, plotFlag)