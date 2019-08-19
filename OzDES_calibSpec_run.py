# ---------------------------------------------------------- #
# ----------------- OzDES_calibSpec_run.py ----------------- #
# ------- https://github.com/jhoormann/OzDES_calibSpec ----- #
# ---------------------------------------------------------- #
# This is a code to perform spectrophotometric calibration.  #
# It was designed to calibrate spectral data from the Anglo  #
# Australian Telescope by matching it to near simultaneous   #
# photometric observations using DECam on the Blanco         #
# Telescope as part of the OzDES Reverberation Mapping       #
# Program.   It also has the option to coadd all spectra     #
# observed either by observing run or by date of observation.#
# The bulk of the calculations are defined in the file       #
# calibSpec_calc.py.  This code defines file locations,      #
# reads in the data, and calls the calibration function.     #
# Unless otherwise noted this code was written by            #
# Janie Hoormann.                                            #
# ---------------------------------------------------------- #
import numpy as np
import OzDES_calibSpec_calc as calc

# First define where all of the data can/will be found

# Define where the transmission function is stored, the bands used, and the centers of each band
bands = [b'g', b'r', b'i']
filters = {b'g': '../OzDES_Pipeline/RMPipeline/input/DES_g_y3a1.dat',
           b'r': '../OzDES_Pipeline/RMPipeline/input/DES_r_y3a1.dat',
           b'i': '../OzDES_Pipeline/RMPipeline/input/DES_i_y3a1.dat'}
centers = [4730, 6420, 7840]

# Define where spectra are stored and file name format: name = spectraBase + ID + spectraEnd
spectraBase = "../OzDES_Data/spectra180413/SVA1_COADD-"
spectraEnd = ".fits"

# Define where photometry are stored and file name format
photoBase = "../OzDES_Data/photometryY5/"
photoEnd = "_lc.dat"

# Define the name of the file that holds the list of sources to calibrate, which we want to be sure is an array
# The OzDES IDs are 10 digit numbers so below, when the variable obj_name is defined it makes sure it was read in as an
# integer and converted to a string.  If your IDs are different be sure to change that too!
idNames = "../OzDES_Data/OzDES_AGN.txt"
names = np.genfromtxt(idNames)

if names.size == 1:
    names = np.array([names])

# Define the name of the place you want the output data stored
outDir = "../Y5_calib_Date/"

# Do you want calibration plots - if so set the flag to the place where they should be saved, otherwise set it to false
plotFlag = False
# plotFlag = "../sparseTest/BBKSparse/"

# Do you want to coadd the spectra? If not the individual calibrated spectra will be save in a fits file
# (coaddFlag == False), otherwise the spectra will be coadded based on the flag chosen (Date: Everything on same mjd
# or Run: Everything on the same observing run)
# coaddFlag = False
coaddFlag = 'Date'
# coaddFlag = 'Run'

# When determining the DES photometric magnitudes at the same time of OzDES spectroscopic light curves the code normally
# just linearly interpolates between the photometry.  This works fine because there is generally such high sampling.
# However, if you have sparser data or what to forecast past when you have data you might want a more robust model.
# You can then use a Gaussian Processes to fit a Brownian Bridge model to the data.  You are allowed to forecast out to
# 28 days.  If you want to change this go to prevent_Excess.
interpFlag = 'linear'
# interpFlag = 'BBK'

# You can also give a file with labeled columns ID and z so the redshift data can be saved with the
# spectra. If you pass through False it will just be saved as -9.99
# redshifts = False
redshifts = "../OzDES_Data/RM_Quasars_z.txt"

# Now we actually call functions and do calculations
for i in range(len(names)):
    obj_name = str(int(names[i]))

    # Define input data names and read in spectra and photometric light curves
    spectraName = spectraBase + obj_name + spectraEnd
    photoName = photoBase + obj_name + photoEnd

    print("Input Spectra Name: %s" % spectraName)
    spectra = calc.Spectrumv18(spectraName)

    # Clean up the spectra.  Marks large isolated large variations in flux and variance as bad (nan) and linearly
    # interpolates over all nans
    calc.mark_as_bad(spectra.flux, spectra.variance)

    print("Input Photometry Name: %s" % photoName)
    photo = np.loadtxt(photoName, dtype={'names':('Date', 'Mag', 'Mag_err', 'Band'),
                                         'formats':(np.float, np.float, np.float, '|S15')}, skiprows=1)

    if redshifts != False:
        zid, red = np.loadtxt("../OzDES_Data/RM_Quasars_z.txt", unpack=True, skiprows=1)
        zid = np.array([str(int(zval)) for zval in zid])

        if obj_name in zid:
            zi = np.where(zid == obj_name)
            redshift = red[zi]
        else:
            redshift = [-9.99]
    else:
        redshift = [-9.99]


    # Calls the main function which does the calibration
    calc.calibSpec(obj_name, spectra, photo, spectraName, photoName, outDir, bands, filters, centers, plotFlag,
                   coaddFlag, interpFlag, redshift)
