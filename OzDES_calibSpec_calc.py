# ---------------------------------------------------------- #
# ----------------- OzDES_calibSpec_calc.py ---------------- #
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

from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import matplotlib.pyplot as plt
import sys


# -------------------------------------------------- #
# Modified from a function originally provided by    #
# Anthea King                                        #
# -------------------------------------------------- #
# ------------------ Spectrumv18 ------------------- #
# -------------------------------------------------- #
# Read in spectral data assuming the format from v18 #
# of the OzDES reduction pipeline. Modify if your    #
# input data is stored differently                   #
# -------------------------------------------------- #

class Spectrumv18(object):
    def __init__(self, filepath=None):
        assert filepath is not None
        self.filepath = filepath
        try:
            self.data = fits.open(filepath)
        except IOError:
            print("Error: file {0} could not be found".format(filepath))
            exit()
        data = fits.open(filepath)
        self.combinedFlux = data[0]
        self.combinedVariance = data[1]
        self.combinedPixels = data[2]
        self.numEpochs = int((np.size(data) - 3) / 3)
        self.field = self.data[3].header['SOURCEF'][19:21]
        self.cdelt1 = self.combinedFlux.header['cdelt1']  # Wavelength interval between subsequent pixels
        self.crpix1 = self.combinedFlux.header['crpix1']
        self.crval1 = self.combinedFlux.header['crval1']
        self.n_pix = self.combinedFlux.header['NAXIS1']
        self.RA = self.combinedFlux.header['RA']
        self.DEC = self.combinedFlux.header['DEC']

        self.fluxCoadd = self.combinedFlux.data
        self.varianceCoadd = self.combinedVariance.data
        self.badpixCoadd = self.combinedPixels.data

        self._wavelength = None
        self._flux = None
        self._variance = None
        self._badpix = None
        self._dates = None
        self._run = None
        self._ext = None
        self._qc = None
        self._exposed = None

    @property
    def wavelength(self):
        """Define wavelength solution."""
        if getattr(self, '_wavelength', None) is None:
            wave = ((np.arange(self.n_pix) - self.crpix1) * self.cdelt1) + self.crval1
            self._wavelength = wave
        return self._wavelength

    @property
    def flux(self):
        if getattr(self, '_flux', None) is None:
            self._flux = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._flux[:, i] = self.data[i * 3 + 3].data
        return self._flux

    @property
    def variance(self):
        if getattr(self, '_variance', None) is None:
            self._variance = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._variance[:, i] = self.data[i * 3 + 4].data
        return self._variance

    @property
    def badpix(self):
        if getattr(self, '_badpix', None) is None:
            self._badpix = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._badpix[:, i] = self.data[i * 3 + 5].data
        return self._badpix

    @property
    def dates(self):
        if getattr(self, '_dates', None) is None:
            self._dates = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._dates[i] = round(self.data[i * 3 + 3].header['UTMJD'],3)
                # this give Modified Julian Date (UTC) that observation was taken
        return self._dates


    @property
    def ext(self):
        if getattr(self, '_ext', None) is None:
            self._ext = []
            for i in range(self.numEpochs):
                self._ext.append(i * 3 + 3)  # gives the extension in original fits file
        return self._ext

    @property
    def run(self):
        if getattr(self, '_run', None) is None:
            self._run = []
            for i in range(self.numEpochs):
                source = self.data[i * 3 + 3].header['SOURCEF']
                self._run.append(int(source[3:6]))  # this gives the run number of the observation
        return self._run

    @property
    def qc(self):
        if getattr(self, '_qc', None) is None:
            self._qc = []
            for i in range(self.numEpochs):
                self._qc.append(self.data[i * 3 + 3].header['QC'])
                # this tell you if there were any problems with the spectra that need to be masked out
        return self._qc

    @property
    def exposed(self):
        if getattr(self, '_exposed', None) is None:
            self._exposed = []
            for i in range(self.numEpochs):
                self._exposed.append(self.data[i * 3 + 3].header['EXPOSED'])
                # this will give you the exposure time of each observation
        return self._exposed


# -------------------------------------------------- #
# ------------------- calibSpec -------------------- #
# -------------------------------------------------- #
# This function does the bulk of the work.  It will  #
# 1) determine extensions which can be calibrated    #
# 2) calculate the scale factors                     #
# 3) calculate the warping function                  #
# 4) output new fits file with scaled spectra        #
# -------------------------------------------------- #

def calibSpec(obj_name, spectra, photo, spectraName, photoName, outBase, bands, filters, centers, plotFlag, coaddFlag,
              interpFlag, redshift):
    # Assumes scaling given is of the form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagErr = scaling[9,:]
    # rMag = scaling[10,:]  rMagErr = scaling[11,:]
    # iMag = scaling[12,:]  iMagErr = scaling[13,:]

    # First we decide which extensions are worth scaling
    extensions, noPhotometry, badQC = prevent_Excess(spectra, photo, bands, interpFlag)

    # Then we calculate the scale factors
    if plotFlag != False:
        plotName = plotFlag + obj_name
    else:
        plotName = False
    nevermind, scaling = scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, bands, filters, interpFlag,
                                        plotName)

    # Remove last minute trouble makers
    extensions = [e for e in extensions if e not in nevermind]
    badQC = badQC + nevermind

    # And finally warp the data
    for s in extensions:
        # scale the spectra
        if plotFlag != False:
            plotName = plotFlag + obj_name + "_" + str(s)
        else:
            plotName = False
        spectra.flux[:, s], spectra.variance[:, s] = warp_spectra(scaling[0:3, s], scaling[3:6, s], spectra.flux[:, s],
                                                                  spectra.variance[:, s], spectra.wavelength, centers,
                                                                  plotName)
    if coaddFlag == False:
        create_output_single(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName,
                             outBase, redshift)
    elif coaddFlag in ['Run', 'Date']:
        coadd_output(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase,
                     plotFlag, coaddFlag, redshift)
    else:
        print("What do you want me to do with this data? Please specify output type.")

    return

# -------------------------------------------------- #
# ---------------- prevent_Excess ------------------ #
# -------------------------------------------------- #
# This function removes extensions from the list to  #
# calibrate because of insufficient photometric data #
# or bad quality flags                               #
# -------------------------------------------------- #

def prevent_Excess(spectra, photo, bands, interpFlag):
    # First, find the min/max date for which we have photometry taken on each side of the spectroscopic observation
    # This will be done by finding the highest date for which we have photometry in each band
    # and taking the max/min of those values
    # This is done because we perform a linear interpolation between photometric data points to estimate the magnitudes
    # observed at the specific time of the spectroscopic observation
    # If you want to use the Gaussian process fitting you can forecast into the future/past by the number of days
    # set by the delay term.

    maxPhot = np.zeros(3)

    # If using Gaussian process fitting you can forecast up to 28 days.  You probably want to make some plots to check
    # this isn't crazy though!
    delay = 0
    if interpFlag == 'BBK':
        delay = 28

    for e in range(len(photo['Date'][:])):
        if photo['Band'][e] == bands[0]:
            if photo['Date'][e] > maxPhot[0]:
                maxPhot[0] = photo['Date'][e]
        if photo['Band'][e] == bands[1]:
            if photo['Date'][e] > maxPhot[1]:
                maxPhot[1] = photo['Date'][e]
        if photo['Band'][e] == bands[2]:
            if photo['Date'][e] > maxPhot[2]:
                maxPhot[2] = photo['Date'][e]
    photLim = min(maxPhot) + delay

    minPhot = np.array([100000, 100000, 100000])
    for e in range(len(photo['Date'][:])):
        if photo['Band'][e] == bands[0]:
            if photo['Date'][e] < minPhot[0]:
                minPhot[0] = photo['Date'][e]
        if photo['Band'][e] == bands[1]:
            if photo['Date'][e] < minPhot[1]:
                minPhot[1] = photo['Date'][e]
        if photo['Band'][e] == bands[2]:
            if photo['Date'][e] < minPhot[2]:
                minPhot[2] = photo['Date'][e]
    photLimMin = max(minPhot) - delay
    noPhotometry = []
    badQC = []

    allowedQC = ['ok', 'backup']

    for s in range(spectra.numEpochs):
        # Remove data with insufficient photometry
        if spectra.dates[s] > photLim:
            noPhotometry.append(s)
        if spectra.dates[s] < photLimMin:
            noPhotometry.append(s)
        # Only allow spectra with quality flags 'ok' and 'backup'
        if spectra.qc[s] not in allowedQC:

            badQC.append(s)

    extensions = []

    # Make a list of extensions which need to be analyzed
    for s in range(spectra.numEpochs):
        if s not in noPhotometry and s not in badQC:
            extensions.append(s)

    return extensions, noPhotometry, badQC

# -------------------------------------------------- #
# ---------------- scaling_Matrix ------------------ #
# -------------------------------------------------- #
# finds the nearest photometry and interpolates mags #
# to find values at the time of the spectroscopic    #
# observations.  Calculates the mag that would be    #
# observed from the spectra and calculates the scale #
# factor to bring them into agreement. Saves the     #
# data in the scaling matrix.                        #
# -------------------------------------------------- #

def scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, bands, filters, interpFlag, plotFlag):
    # scale factors for each extension saved in the following form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagError = scaling[9,:] (interpolated from neighbouring observations)
    # rMag = scaling[10,:]   rMagError = scaling[11,:]
    # iMag = scaling[12,:]   iMagError = scaling[13,:]

    scaling = np.zeros((14, spectra.numEpochs))

    # Judge goodness of spectra
    for e in range(spectra.numEpochs):
        if e in badQC:
            scaling[6, e] = False
        else:
            scaling[6, e] = True
        if e in noPhotometry:
            scaling[7, e] = False
        else:
            scaling[7, e] = True

    ozdesPhoto = np.zeros((3, spectra.numEpochs))
    desPhoto = np.zeros((3, spectra.numEpochs))

    ozdesPhotoU = np.zeros((3, spectra.numEpochs))
    desPhotoU = np.zeros((3, spectra.numEpochs))

    filterCurves = readFilterCurves(bands, filters)

    if interpFlag == 'BBK':
        desPhoto, desPhotoU = des_photo_BBK(photo, spectra.dates, bands, spectra.numEpochs, plotFlag)

        scaling[8, :] = desPhoto[0, :]
        scaling[10, :] = desPhoto[1, :]
        scaling[12, :] = desPhoto[2, :]

        scaling[9, :] = desPhotoU[0, :]
        scaling[11, :] = desPhotoU[1, :]
        scaling[13, :] = desPhotoU[2, :]

    nevermind = []

    for e in extensions:
        # Find OzDES photometry

        ozdesPhoto[0, e], ozdesPhotoU[0, e] = computeABmag(filterCurves[bands[0]].trans, filterCurves[bands[0]].wave,
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[1, e], ozdesPhotoU[1, e] = computeABmag(filterCurves[bands[1]].trans, filterCurves[bands[1]].wave,
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[2, e], ozdesPhotoU[2, e] = computeABmag(filterCurves[bands[2]].trans, filterCurves[bands[2]].wave,
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])

        # Sometimes the total flux in the band goes zero and this obviously creates issues further down the line and
        # is most noticeable when the calculated magnitude is nan.  Sometimes it is because the data is very noisy
        # or the occasional negative spectrum is a known artifact of the data, more common in early OzDES runs.  In the
        # case where the observation doesn't get cut based on quality flag it will start getting ignored here.  The runs
        # ignored will eventually be saved with the badQC extensions.

        if np.isnan(ozdesPhoto[:, e]).any() == True:
            nevermind.append(e)

        # Find DES photometry
        if interpFlag == 'linear':
            desPhoto[:, e], desPhotoU[:, e] = des_photo(photo, spectra.dates[e], bands)

            scaling[8, e] = desPhoto[0, e]
            scaling[10, e] = desPhoto[1, e]
            scaling[12, e] = desPhoto[2, e]

            scaling[9, e] = desPhotoU[0, e]
            scaling[11, e] = desPhotoU[1, e]
            scaling[13, e] = desPhotoU[2, e]

        # Find Scale Factor
        scaling[0, e], scaling[3, e] = scale_factors(desPhoto[0, e] - ozdesPhoto[0, e],
                                                     desPhotoU[0, e] + ozdesPhotoU[0, e])
        scaling[1, e], scaling[4, e] = scale_factors(desPhoto[1, e] - ozdesPhoto[1, e],
                                                     desPhotoU[1, e] + ozdesPhotoU[1, e])
        scaling[2, e], scaling[5, e] = scale_factors(desPhoto[2, e] - ozdesPhoto[2, e],
                                                     desPhotoU[2, e] + ozdesPhotoU[2, e])


    return nevermind, scaling

# -------------------------------------------------- #
# The next three functions are modified from code    #
# provided by Dale Mudd                              #
# -------------------------------------------------- #
# ------------------ filterCurve ------------------- #
# -------------------------------------------------- #
# creates a class to hold the transmission function  #
# for each band.                                     #
# -------------------------------------------------- #

class filterCurve:
    """A filter"""

    def __init__(self):
        self.wave = np.array([], 'float')
        self.trans = np.array([], 'float')
        return

    def read(self, file):
        # DES filter curves express the wavelengths in nms
        if 'DES' in file:
            factor = 10.
        else:
            factor = 1.
        file = open(file, 'r')
        for line in file.readlines():
            if line[0] != '#':
                entries = line.split()
                self.wave = np.append(self.wave, float(entries[0]))
                self.trans = np.append(self.trans, float(entries[1]))
        file.close()
        # We use Angstroms for the wavelength in the filter transmission file
        self.wave = self.wave * factor
        return

# -------------------------------------------------- #
# ---------------- readFilterCurve ----------------- #
# -------------------------------------------------- #
# Reads in the filter curves and stores it as the    #
# filter curve class.                                #
# -------------------------------------------------- #

def readFilterCurves(bands, filters):

    filterCurves = {}
    for f in bands:
        filterCurves[f] = filterCurve()
        filterCurves[f].read(filters[f])

    return filterCurves

# -------------------------------------------------- #
# ----------------- computeABmag ------------------- #
# -------------------------------------------------- #
# computes the AB magnitude for given transmission   #
# functions and spectrum (f_lambda).  Returns the    #
# magnitude and variance.                            #
# -------------------------------------------------- #

def computeABmag(trans_flux, trans_wave, tmp_wave, tmp_flux, tmp_var):
    # Takes and returns variance
    # trans_ : transmission function data
    # tmp_ : spectral data

    # trans/tmp not necessarily defined over the same wavelength range
    # first determine the wavelength range over which both are defined
    minV = min(trans_wave)
    if minV < min(tmp_wave):
        minV = min(tmp_wave)
    maxV = max(trans_wave)
    if maxV > max(trans_wave):
        maxV = max(trans_wave)

    interp_wave = []
    tmp_flux2 = []
    tmp_var2 = []

    # Make new vectors for the flux just using that range (assuming spectral binning)

    for i in range(len(tmp_wave)):
        if minV < tmp_wave[i] < maxV:
            interp_wave.append(tmp_wave[i])
            tmp_flux2.append(tmp_flux[i])
            tmp_var2.append(tmp_var[i])

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_flux2 = interp1d(trans_wave, trans_flux)(interp_wave)

    # And now calculate the magnitude and uncertainty

    c = 2.992792e18  # Angstrom/s
    Num = np.nansum(tmp_flux2 * trans_flux2 * interp_wave)
    Num_var = np.nansum(tmp_var2 * (trans_flux2 * interp_wave) ** 2)
    Den = np.nansum(trans_flux2 / interp_wave)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
            magABvar = 1.17882 * Num_var / (Num ** 2)
        except FloatingPointError:
            magAB = 99.
            magABvar = 99.

    return magAB, magABvar

# -------------------------------------------------- #
# ------------------ des_photo  -------------------- #
# -------------------------------------------------- #
# Finds nearest photometry on both sides of spectral #
# observations and interpolates to find value at the #
# time of the spectral observation.                  #
# -------------------------------------------------- #

def des_photo(photo, spectral_mjd, bands):

    """Takes in an mjd from the spectra, looks through a light curve file to find the nearest photometric epochs and
    performs linear interpolation to get estimate at date, return the photo mags.   """

    # Assumes dates are in chronological order!!!
    mags = np.zeros(3)
    errs = np.zeros(3)

    for l in range(len(photo['Date']) - 1):
        if photo['Band'][l] == bands[0] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            g_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            g_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            g_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])
        if photo['Band'][l] == bands[1] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            r_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            r_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            r_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])
        if photo['Band'][l] == bands[2] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            i_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            i_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            i_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])

    mags[0], errs[0] = interpolatePhot(g_date_v, g_mag_v, g_err_v, spectral_mjd)
    mags[1], errs[1] = interpolatePhot(r_date_v, r_mag_v, r_err_v, spectral_mjd)
    mags[2], errs[2] = interpolatePhot(i_date_v, i_mag_v, i_err_v, spectral_mjd)

    return mags, errs


# -------------------------------------------------- #
# ---------------- des_photo_BBK  ------------------ #
# -------------------------------------------------- #
# Finds nearest photometry on both sides of spectral #
# observations and interpolates to find value at the #
# time of the spectral observations using Brownian   #
# Bridge Gaussian processes.  This is better for     #
# sparser data.                                      #
# -------------------------------------------------- #

def des_photo_BBK(photo, dates, bands, numEpochs, plotFlag):

    # Assumes dates are in chronological order!!!
    mags = np.zeros((3, numEpochs))

    errs = np.zeros((3, numEpochs))

    # Fit a Brownian Bridge Kernel to the data via Gaussian processes.
    for b in range(3):
        x = []  # Dates for each band
        y = []  # Mags for each band
        s = []  # Errors for each band

        # get data for each band
        for l in range(len(photo['Date']) - 1):
            if photo['Band'][l] == bands[b]:
                x.append(photo['Date'][l])
                y.append(photo['Mag'][l])
                s.append(photo['Mag_err'][l])

        x = np.array(x)
        y = np.array(y)
        s = np.array(s)

        # Define kernel for Gaussian process: Browning Bridge x Constant
        kernel1 = BBK(length_scale=25, length_scale_bounds=(1, 1000))
        kernel2 = kernels.ConstantKernel(constant_value=1.0, constant_value_bounds=(0.001, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel1 * kernel2, alpha=s ** 2, normalize_y=True)

        # Fit the data with the model
        xprime = np.atleast_2d(x).T
        yprime = np.atleast_2d(y).T
        gp.fit(xprime, yprime)

        if plotFlag != False:
            # Plot what the model looks like
            bname = ['_g', '_r', '_i']
            preddates = np.linspace(min(x) - 100, max(x) + 100, 3000)
            y_predAll, sigmaAll = gp.predict(np.atleast_2d(preddates).T, return_std=True)
            y_predAll = y_predAll.flatten()
            fig, ax1 = makeFigSingle(plotFlag + bname[b], 'Date', 'Mag', [dates[0], dates[-1]])

            # I want to plot lines where the observations take place - only plot one per night though
            dateCull = dates.astype(int)
            dateCull = np.unique(dateCull)
            for e in range(len(dateCull)):
                ax1.axvline(dateCull[e], color='grey', alpha=0.5)
            ax1.errorbar(x, y, yerr=s, fmt='o', color='mediumblue', markersize='7')

            # Plot model with error bars.
            ax1.plot(preddates, y_predAll, color='black')
            ax1.fill_between(preddates, y_predAll - sigmaAll, y_predAll + sigmaAll, alpha=0.5, color='black')
            plt.savefig(plotFlag + bname[b] + "_photoModel.png")
            plt.close(fig)

        # Predict photometry vales for each observation
        y_pred, sigma = gp.predict(np.atleast_2d(dates).T, return_std=True)
        mags[b, :] = y_pred.flatten()
        errs[b, :] = sigma[0]**2

    return mags, errs

# -------------------------------------------------- #
# --------------- interpolatePhot  ----------------- #
# -------------------------------------------------- #
# Performs linear interpolation and propagates the   #
# uncertainty to return you a variance.              #
# -------------------------------------------------- #

def interpolatePhot(x, y, s, val):
    # takes sigma returns variance
    # x - x data points (list)
    # y - y data points (list)
    # s - sigma on y data points (list)
    # val - x value to interpolate to (number)

    mag = y[0] + (val - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

    err = s[0] ** 2 + (s[0] ** 2 + s[1] ** 2) * ((val - x[0]) / (x[1] - x[0])) ** 2

    return mag, err

# -------------------------------------------------- #
# ---------------- scale_factors  ------------------ #
# -------------------------------------------------- #
# Calculates the scale factor and variance needed to #
# change spectroscopically derived magnitude to the  #
# observed photometry.                               #
# -------------------------------------------------- #

def scale_factors(mag_diff, mag_diff_var):
    # takes and returns variance

    flux_ratio = np.power(10., 0.4 * mag_diff)  # f_synthetic/f_photometry
    scale_factor = (1. / flux_ratio)
    scale_factor_sigma = mag_diff_var * (scale_factor * 0.4 * 2.3) ** 2   # ln(10) ~ 2.3

    return scale_factor, scale_factor_sigma

# -------------------------------------------------- #
# ----------------- warp_spectra  ------------------ #
# -------------------------------------------------- #
# Fits polynomial to scale factors and estimates     #
# associated uncertainties with gaussian processes.  #
# If the plotFlag variable is not False it will save #
# some diagnostic plots.                             #
# -------------------------------------------------- #

def warp_spectra(scaling, scaleErr, flux, variance, wavelength, centers, plotFlag):

    # associate scale factors with centers of bands and fit 2D polynomial to form scale function.
    scale = InterpolatedUnivariateSpline(centers, scaling, k=2)
    fluxScale = flux * scale(wavelength)

    # add in Gaussian process to estimate uncertainties, /10**-17 because it gets a bit panicky if you use small numbers
    stddev = (scaleErr ** 0.5) / 10 ** -17
    scale_v = scaling / 10 ** -17

    kernel = kernels.RBF(length_scale=300, length_scale_bounds=(.01, 2000.0))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=stddev**2)

    xprime = np.atleast_2d(centers).T
    yprime = np.atleast_2d(scale_v).T

    gp.fit(xprime, yprime)
    xplot_prime = np.atleast_2d(wavelength).T
    y_pred, sigma = gp.predict(xplot_prime, return_std=True)

    y_pred = y_pred[:,0]

    sigModel = (sigma/y_pred)*scale(wavelength)

    # now scale the original variance and combine with scale factor uncertainty
    varScale = variance * pow(scale(wavelength), 2) + sigModel ** 2

    # if plotFlag != False:
    #     figa, ax1a, ax2a = makeFigDouble(plotFlag, "Wavelength ($\AA$)", "f$_\lambda$ (arbitrary units)",
    #                                   "f$_\lambda$ (10$^{-17}$ erg/s/cm$^2$/$\AA$)", [wavelength[0], wavelength[-1]])
    #
    #     ax1a.plot(wavelength, flux, color='black', label="Before Calibration")
    #     ax1a.legend(loc=1, frameon=False, prop={'size': 20})
    #     ax2a.plot(wavelength, fluxScale / 10 ** -17, color='black', label="After Calibration")
    #     ax2a.legend(loc=1, frameon=False, prop={'size': 20})
    #     plt.savefig(plotFlag + "_beforeAfter.png")
    #     plt.close(figa)
    #
    #     figb, ax1b, ax2b = makeFigDouble(plotFlag, "Wavelength ($\AA$)", "f$_\lambda$ (10$^{-17}$ erg/s/cm$^2$/$\AA$)",
    #                                      "% Uncertainty", [wavelength[0], wavelength[-1]])
    #     ax1b.plot(wavelength, fluxScale / 10 ** -17, color='black')
    #
    #     ax2b.plot(wavelength, 100*abs(pow(varScale, 0.5)/fluxScale), color='black', linestyle='-', label='Total')
    #     ax2b.plot(wavelength, 100*abs(sigModel/fluxScale), color='blue', linestyle='-.', label='Warping')
    #     ax2b.legend(loc=1, frameon=False, prop={'size': 20})
    #     ax2b.set_ylim([0, 50])
    #     plt.savefig(plotFlag + "_uncertainty.png")
    #     plt.close(figb)
    #
    #     figc, axc = makeFigSingle(plotFlag, "Wavelength ($\AA$)", "Scale Factor (10$^{-17}$ erg/s/cm$^2$/$\AA$/counts)")
    #     axc.plot(wavelength, scale(wavelength)/10**-17, color='black')
    #     axc.errorbar(centers, scaling/10**-17, yerr=stddev, fmt='s', color='mediumblue')
    #     plt.savefig(plotFlag + "_scalefactors.png")
    #     plt.close(figc)

    return fluxScale, varScale


# -------------------------------------------------- #
# ------------ create_output_single  --------------- #
# -------------------------------------------------- #
# Outputs the warped spectra to a new fits file.     #
# -------------------------------------------------- #
def create_output_single(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase,
                         redshift):

    outName = outBase + obj_name + "_scaled.fits"
    print("Saving Data to " + outName)

    hdulist = fits.HDUList(fits.PrimaryHDU())

    noPhotometryExt = []
    if len(noPhotometry) > 0:
        for i in range(len(noPhotometry)):
            noPhotometryExt.append(spectra.ext[noPhotometry[i]])

    badQCExt = []
    if len(badQC) > 0:
        for i in range(len(badQC)):
            badQCExt.append(spectra.ext[badQC[i]])

    index = 0
    # Create an HDU for each night
    for i in extensions:
        header = fits.Header()
        header['SOURCE'] = obj_name
        header['RA'] = spectra.RA
        header['DEC'] = spectra.DEC
        header['FIELD'] = spectra.field
        header['CRPIX1'] = spectra.crpix1
        header['CRVAL1'] = spectra.crval1
        header['CDELT1'] = spectra.cdelt1
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['EPOCHS'] = len(extensions)
        header['z'] = redshift[0]

        # save the names of the input data and the extensions ignored
        header['SFILE'] = spectraName
        header['PFILE'] = photoName
        header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
        header['BADQC'] = ','.join(map(str, badQCExt))

        # save the original spectrum's extension number and some other details
        header["EXT"] = spectra.ext[i]
        header["UTMJD"] = spectra.dates[i]
        header["EXPOSE"] = spectra.exposed[i]
        header["QC"] = spectra.qc[i]

        # save scale factors/uncertainties
        header["SCALEG"] = scaling[0, i]
        header["ERRORG"] = scaling[3, i]
        header["SCALER"] = scaling[1, i]
        header["ERRORR"] = scaling[4, i]
        header["SCALEI"] = scaling[2, i]
        header["ERRORI"] = scaling[5, i]

        # save photometry/uncertainties used to calculate scale factors
        header["MAGG"] = scaling[8, i]
        header["MAGUG"] = scaling[9, i]
        header["MAGR"] = scaling[10, i]
        header["MAGUR"] = scaling[11, i]
        header["MAGI"] = scaling[12, i]
        header["MAGUI"] = scaling[13, i]
        if index == 0:
            hdulist[0].header['SOURCE'] = obj_name
            hdulist[0].header['RA'] = spectra.RA
            hdulist[0].header['DEC'] = spectra.DEC
            hdulist[0].header['CRPIX1'] = spectra.crpix1
            hdulist[0].header['CRVAL1'] = spectra.crval1
            hdulist[0].header['CDELT1'] = spectra.cdelt1
            hdulist[0].header['CTYPE1'] = 'wavelength'
            hdulist[0].header['CUNIT1'] = 'angstrom'
            hdulist[0].header['EPOCHS'] = len(extensions)

            # save the names of the input data and the extensions ignored
            hdulist[0].header['SFILE'] = spectraName
            hdulist[0].header['PFILE'] = photoName
            hdulist[0].header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
            hdulist[0].header['BADQC'] = ','.join(map(str, badQCExt))

            # save the original spectrum's extension number and some other details
            hdulist[0].header["EXT"] = spectra.ext[i]
            hdulist[0].header["UTMJD"] = spectra.dates[i]
            hdulist[0].header["EXPOSE"] = spectra.exposed[i]
            hdulist[0].header["QC"] = spectra.qc[i]

            # save scale factors/uncertainties
            hdulist[0].header["SCALEG"] = scaling[0, i]
            hdulist[0].header["ERRORG"] = scaling[3, i]
            hdulist[0].header["SCALER"] = scaling[1, i]
            hdulist[0].header["ERRORR"] = scaling[4, i]
            hdulist[0].header["SCALEI"] = scaling[2, i]
            hdulist[0].header["ERRORI"] = scaling[5, i]

            # save photometry/uncertainties used to calculate scale factors
            hdulist[0].header["MAGG"] = scaling[8, i]
            hdulist[0].header["MAGUG"] = scaling[9, i]
            hdulist[0].header["MAGR"] = scaling[10, i]
            hdulist[0].header["MAGUR"] = scaling[11, i]
            hdulist[0].header["MAGI"] = scaling[12, i]
            hdulist[0].header["MAGUI"] = scaling[13, i]
            hdulist[0].data = spectra.flux[:, i]
            hdulist.append(fits.ImageHDU(data=spectra.variance[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.badpix[:, i], header=header))
            index = 2


        else:
            hdulist.append(fits.ImageHDU(data=spectra.flux[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.variance[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.badpix[:, i], header=header))
    hdulist.writeto(outName, overwrite=True)
    hdulist.close()

    return

# -------------------------------------------------- #
# ------------- create_output_coadd  --------------- #
# -------------------------------------------------- #
# Outputs the warped and coadded spectra to a new    #
# fits file.                                         #
# -------------------------------------------------- #


def create_output_coadd(obj_name, runList, fluxArray, varianceArray, badpixArray, extensions, scaling, spectra, redshift
                        ,badQC, noPhotometry, spectraName, photoName, outBase, coaddFlag):

    outName = outBase + obj_name + "_scaled_" + coaddFlag + ".fits"
    hdulist = fits.HDUList(fits.PrimaryHDU())

    noPhotometryExt = []
    if len(noPhotometry) > 0:
        for i in range(len(noPhotometry)):
            noPhotometryExt.append(spectra.ext[noPhotometry[i]])

    badQCExt = []
    if len(badQC) > 0:
        for i in range(len(badQC)):
            badQCExt.append(spectra.ext[badQC[i]])

    #print("Output Filename: %s \n" % (outName))
    # First save the total coadded spectrum for the source to the primary extension
    hdulist[0].data = fluxArray[:, 0]
    hdulist[0].header['CRPIX1'] = spectra.crpix1
    hdulist[0].header['CRVAL1'] = spectra.crval1
    hdulist[0].header['CDELT1'] = spectra.cdelt1
    hdulist[0].header['CTYPE1'] = 'wavelength'
    hdulist[0].header['CUNIT1'] = 'angstrom'
    hdulist[0].header['SOURCE'] = obj_name
    hdulist[0].header['RA'] = spectra.RA
    hdulist[0].header['DEC'] = spectra.DEC
    hdulist[0].header['FIELD'] = spectra.field
    hdulist[0].header['OBSNUM'] = len(runList)
    hdulist[0].header['z'] = redshift[0]
    hdulist[0].header['SFILE'] = spectraName
    hdulist[0].header['PFILE'] = photoName
    hdulist[0].header['METHOD'] = coaddFlag
    hdulist[0].header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
    hdulist[0].header['BADQC'] = ','.join(map(str, badQCExt))

    # First extension is the total coadded variance
    header = fits.Header()
    header['EXTNAME'] = 'VARIANCE'
    header['CRPIX1'] = spectra.crpix1
    header['CRVAL1'] = spectra.crval1
    header['CDELT1'] = spectra.cdelt1
    header['CTYPE1'] = 'wavelength'
    header['CUNIT1'] = 'angstrom'
    hdulist.append(fits.ImageHDU(data=varianceArray[:, 0], header=header))

    # Second Extension is the total bad pixel map
    header = fits.Header()
    header['EXTNAME'] = 'BadPix'
    header['CRPIX1'] = spectra.crpix1
    header['CRVAL1'] = spectra.crval1
    header['CDELT1'] = spectra.cdelt1
    header['CTYPE1'] = 'wavelength'
    header['CUNIT1'] = 'angstrom'
    hdulist.append(fits.ImageHDU(data=badpixArray[:, 0], header=header))

    # Create an HDU for each night
    index1 = 1
    for k in runList:
        index = 0
        date = 0
        header = fits.Header()
        header['CRPIX1'] = spectra.crpix1
        header['CRVAL1'] = spectra.crval1
        header['CDELT1'] = spectra.cdelt1
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['RUN'] = k
        for i in extensions:
            here = False
            if coaddFlag == 'Run':
                if spectra.run[i] == k:
                    here = True

            if coaddFlag == 'Date':
                if int(spectra.dates[i]) == k:
                    here = True

            if here == True:
                head0 = "EXT" + str(index)
                header[head0] = spectra.ext[i]

                head1 = "UTMJD" + str(index)
                header[head1] = spectra.dates[i]
                date += spectra.dates[i]

                head2 = "EXPOSE" + str(index)
                header[head2] = spectra.exposed[i]

                head3 = "QC" + str(index)
                header[head3] = spectra.qc[i]

                head4 = "SCALEG" + str(index)
                header[head4] = scaling[0, i]

                head5 = "ERRORG" + str(index)
                header[head5] = scaling[3, i]

                head6 = "SCALER" + str(index)
                header[head6] = scaling[1, i]

                head7 = "ERRORR" + str(index)
                header[head7] = scaling[4, i]

                head8 = "SCALEI" + str(index)
                header[head8] = scaling[2, i]

                head9 = "ERRORI" + str(index)
                header[head9] = scaling[5, i]

                head10 = "MAGG" + str(index)
                header[head10] = scaling[8, i]

                head11 = "MAGUG" + str(index)
                header[head11] = scaling[9, i]

                head12 = "MAGR" + str(index)
                header[head12] = scaling[10, i]

                head13 = "MAGUR" + str(index)
                header[head13] = scaling[11, i]

                head14 = "MAGI" + str(index)
                header[head14] = scaling[12, i]

                head15 = "MAGUI" + str(index)
                header[head15] = scaling[13, i]

                index += 1

        if date > 0:
            header['OBSNUM'] = index
            header['AVGDATE'] = date / index

            hdu_flux = fits.ImageHDU(data=fluxArray[:, index1], header=header)
            hdu_fluxvar = fits.ImageHDU(data=varianceArray[:, index1], header=header)
            hdu_badpix = fits.ImageHDU(data=badpixArray[:, index1], header=header)
            hdulist.append(hdu_flux)
            hdulist.append(hdu_fluxvar)
            hdulist.append(hdu_badpix)
        index1 += 1

    hdulist.writeto(outName, overwrite=True)
    hdulist.close()

    return

# -------------------------------------------------- #
# ----------------- coadd_output  ------------------ #
# -------------------------------------------------- #
# Coadds the observations based on run or night.     #
# -------------------------------------------------- #
def coadd_output(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase, plotFlag,
                 coaddFlag, redshift):

    # Get a list of items (dates/runs) over which all observations will be coadded
    coaddOver = []

    for e in extensions:
        # OzDES runs 7,8 were close together in time and run 8 had bad weather so there was only observations of 1
        # field - coadd with run 7 to get better signal to noise
        if spectra.run[e] == 8:
            spectra.run[e] = 7

        if coaddFlag == 'Run':
            if spectra.run[e] not in coaddOver:
                coaddOver.append(spectra.run[e])

        if coaddFlag == 'Date':
            if int(spectra.dates[e]) not in coaddOver:
                coaddOver.append(int(spectra.dates[e]))


    coaddFlux = np.zeros((5000, len(coaddOver) + 1))
    coaddVar = np.zeros((5000, len(coaddOver) + 1))
    coaddBadPix = np.zeros((5000, len(coaddOver) + 1))

    speclistC = []  # For total coadd of observation
    index = 1

    for c in coaddOver:
        speclist = []
        for e in extensions:
            opt = ''
            if coaddFlag == 'Run':
                opt = spectra.run[e]
            if coaddFlag == 'Date':
                opt = int(spectra.dates[e])
            if opt == c:
                speclist.append(SingleSpec(obj_name, spectra.wavelength, spectra.flux[:,e], spectra.variance[:,e],
                                           spectra.badpix[:,e]))
                speclistC.append(SingleSpec(obj_name, spectra.wavelength, spectra.flux[:,e], spectra.variance[:,e],
                                            spectra.badpix[:,e]))

        if len(speclist) > 1:
            runCoadd = outlier_reject_and_coadd(obj_name, speclist)
            coaddFlux[:, index] = runCoadd.flux
            coaddVar[:, index] = runCoadd.fluxvar
            coaddVar[:, index] = runCoadd.fluxvar
            coaddBadPix[:,index] = runCoadd.isbad.astype('uint8')
        if len(speclist) == 1:
            coaddFlux[:, index] = speclist[0].flux
            coaddVar[:, index] = speclist[0].fluxvar
            coaddBadPix[:, index] = speclist[0].isbad.astype('uint8')
        index += 1

    if len(speclistC) > 1:
        allCoadd = outlier_reject_and_coadd(obj_name, speclistC)
        coaddFlux[:, 0] = allCoadd.flux
        coaddVar[:, 0] = allCoadd.fluxvar
        coaddBadPix[:, 0] = allCoadd.isbad.astype('uint8')
    if len(speclistC) == 1:
        coaddFlux[:, 0] = speclistC[0].flux
        coaddVar[:, 0] = speclistC[0].fluxvar
        coaddBadPix[:, 0] = speclistC[0].isbad.astype('uint8')

    mark_as_bad(coaddFlux, coaddVar)

    create_output_coadd(obj_name, coaddOver, coaddFlux, coaddVar, coaddBadPix, extensions, scaling, spectra, redshift,
                        badQC, noPhotometry, spectraName, photoName, outBase, coaddFlag)


    return

# -------------------------------------------------- #
# Modified from code originally provided by          #
# Harry Hobson                                       #
# -------------------------------------------------- #
# ------------------ mark_as_bad ------------------- #
# -------------------------------------------------- #
# Occasionally you get some big spikes in the data   #
# that you do not want messing with your magnitude   #
# calculations.  Remove these by looking at single   #
# bins that have a significantly 4.5 larger than     #
# average fluxes or variances and change those to    #
# nans. Nans will be interpolated over.  The         #
# threshold should be chosen to weigh removing       #
# extreme outliers and removing noise.               #
# -------------------------------------------------- #

def mark_as_bad(fluxes, variances):
    number = int(fluxes.size/fluxes.shape[0])
    for epoch in range(number):
        if number == 1:
            flux = fluxes[:]
            variance = variances[:]
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        nBins = len(flux)
        # define the local average in flux and variance to compare outliers to
        for i in range(nBins):
            if i < 50:
                avg = np.nanmean(variance[0:99])
                avgf = np.nanmean(flux[0:99])
            elif i > nBins - 50:
                avg = np.nanmean(variance[i-50:nBins-1])
                avgf = np.nanmean(flux[i-50:nBins-1])
            else:
                avg = np.nanmean(variance[i-50:i+50])
                avgf = np.nanmean(flux[i-50:i+50])

            # find outliers and set that bin and the neighbouring ones to nan.

            if np.isnan(variance[i]) == False and variance[i] > 4.5*avg:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i - 1] = np.nan
                    flux[i - 2] = np.nan
                    flux[i - 3] = np.nan
                    flux[i + 1] = np.nan
                    flux[i + 2] = np.nan
                    flux[i + 3] = np.nan

            if np.isnan(flux[i]) == False and flux[i] > 4.5 * avgf:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan

            if np.isnan(flux[i]) == False and flux[i] < -4.5 * avgf:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan

        # interpolates nans (added here and bad pixels in the data)
        filter_bad_pixels(flux, variance)
    return


# -------------------------------------------------- #
# Modified from code originally provided by          #
# Harry Hobson                                       #
# -------------------------------------------------- #
# --------------- filter_bad_pixels ---------------- #
# -------------------------------------------------- #
# Interpolates over nans in the spectrum.            #
# -------------------------------------------------- #
def filter_bad_pixels(fluxes, variances):
    number = int(fluxes.size/fluxes.shape[0])
    for epoch in range(number):
        if (number == 1):
            flux = fluxes[:]
            variance = variances[:]
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        nBins = len(flux)

        flux[0] = np.nanmean(flux)/1000
        flux[-1] = np.nanmean(flux)/1000
        variance[0] = 100*np.nanmean(variance)
        variance[-1] = 100*np.nanmean(variance)

        bad_pixels = np.logical_or.reduce((np.isnan(flux), np.isnan(variance), variance < 0))

        bin = 0
        binEnd = 0

        while (bin < nBins):
            if (bad_pixels[bin] == True):
                binStart = bin
                binNext = bin + 1
                while (binNext < nBins):
                    if bad_pixels[binNext] == False:
                        binEnd = binNext - 1
                        binNext = nBins
                    binNext = binNext + 1

                ya = float(flux[binStart - 1])
                xa = float(binStart - 1)
                sa = variance[binStart - 1]
                yb = flux[binEnd + 1]
                xb = binEnd + 1
                sb = variance[binEnd + 1]

                step = binStart
                while (step < binEnd + 1):
                    flux[step] = ya + (yb - ya) * (step - xa) / (xb - xa)
                    variance[step] = sa + (sb + sa) * ((step - xa) / (xb - xa)) ** 2
                    step = step + 1
                bin = binEnd
            bin = bin + 1
    return


# -------------------------------------------------- #
# ----------------- makeFigDouble ------------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A function that defines a figure and axes with two #
# panels that shares an x axis and has legible axis  #
# labels.                                            #
# -------------------------------------------------- #
font = {'size': '20', 'color': 'black', 'weight': 'normal'}

def makeFigDouble(title, xlabel, ylabel1, ylabel2, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0]):

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)
    fig.subplots_adjust(hspace=0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(20)

    ax1.set_ylabel(ylabel1, **font)
    if ylim1 != [0, 0] and ylim1[0] < ylim1[1]:
        ax1.set_ylim(ylim1)

    ax2.set_ylabel(ylabel2, **font)
    if ylim2 != [0, 0] and ylim2[0] < ylim2[1]:
        ax2.set_ylim(ylim2)

    ax2.set_xlabel(xlabel, **font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax2.set_xlim(xlim)

    ax1.set_title(title, **font)

    return fig, ax1, ax2

# -------------------------------------------------- #
# ----------------- makeFigSingle ------------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A function that defines a figure with legible axis #
# labels.                                            #
# -------------------------------------------------- #
def makeFigSingle(title, xlabel, ylabel, xlim=[0, 0], ylim=[0, 0]):
    fig = plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)

    ax = fig.add_subplot(111)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    ax.set_ylabel(ylabel, **font)
    if ylim != [0, 0] and ylim[0] < ylim[1]:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, **font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax.set_xlim(xlim)

    ax.set_title(title, **font)

    return fig, ax


# -------------------------------------------------- #
#  The following 4 functions were written by Chris   #
# Lidman, Mike Childress, and maybe others for the   #
# initial processing of the OzDES spectra.  They     #
# were taken from the DES_coaddSpectra.py functions. #
# -------------------------------------------------- #
# -------------------- OzExcept -------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A simple exception class                           #
# -------------------------------------------------- #


class OzExcept(Exception):
    """
    Simple exception class
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "{0}: {1}".format(self.__class__.__name__, msg)


# -------------------------------------------------- #
# ----------------- VerboseMessager ---------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# Verbose messaging for routines below.              #
# -------------------------------------------------- #


class VerboseMessager(object):
    """
    Verbose messaging for routines below
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, *args):
        if self.verbose:
            print("Something strange is happening")
            sys.stdout.flush()

# -------------------------------------------------- #
# ------------------- SingleSpec ------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# Class representing a single spectrum for analysis. #
# -------------------------------------------------- #
class SingleSpec(object):
    """
    Class representing a single spectrum for analysis
    """

    ## Added filename to SingleSpec
    def __init__(self, obj_name, wl, flux, fluxvar, badpix):

        self.name = obj_name
        # ---------------------------
        # self.pivot = int(fibrow[9])
        # self.xplate = int(fibrow[3])
        # self.yplate = int(fibrow[4])
        # self.ra = np.degrees(fibrow[1])
        # self.dec = np.degrees(fibrow[2])
        # self.mag=float(fibrow[10])
        # self.header=header

        self.wl = np.array(wl)
        self.flux = np.array(flux)
        self.fluxvar = np.array(fluxvar)

        # If there is a nan in either the flux, or the variance, mark it as bad

        # JKH: this was what was here originally, my version complains about it
        # self.fluxvar[fluxvar < 0] = np.nan

        for i in range(5000):
            if (self.fluxvar[i] < 0):
                self.fluxvar[i] = np.nan

        # The following doesn't take into account
        #self.isbad = np.any([np.isnan(self.flux), np.isnan(self.fluxvar)], axis=0)
        self.isbad = badpix.astype(bool)

# -------------------------------------------------- #
# ------------ outlier_reject_and_coadd ------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #
# OzDES coadding function to reject outliers and     #
# coadd all of the spectra in the inputted list.     #
# -------------------------------------------------- #
def outlier_reject_and_coadd(obj_name, speclist):
    """
    Reject outliers on single-object spectra to be coadded.
    Assumes input spectra have been resampled to a common wavelength grid,
    so this step needs to be done after joining and resampling.

    Inputs
        speclist:  list of SingleSpec instances on a common wavelength grid
        show:  boolean; show diagnostic plot?  (debug only; default=False)
        savefig:  boolean; save diagnostic plot?  (debug only; default=False)
    Output
        result:  SingleSpec instance of coadded spectrum, with bad pixels
            set to np.nan (runz requires this)
    """

    # Edge cases
    if len(speclist) == 0:
        print("outlier_reject:  empty spectrum list")
        return None
    elif len(speclist) == 1:
        tgname = speclist[0].name
        vmsg("Only one spectrum, no coadd needed for {0}".format(tgname))
        return speclist[0]

    # Have at least two spectra, so let's try to reject outliers
    # At this stage, all spectra have been mapped to a common wavelength scale
    wl = speclist[0].wl
    tgname = speclist[0].name
    # Retrieve single-object spectra and variance spectra.
    flux_2d = np.array([s.flux for s in speclist])
    fluxvar_2d = np.array([s.fluxvar for s in speclist])
    badpix_2d = np.array([s.isbad for s in speclist])


    # Baseline parameters:
    #    outsig     Significance threshold for outliers (in sigma)
    #    nbin       Bin width for median rebinning
    #    ncoinc     Maximum number of spectra in which an artifact can appear
    outsig, nbin, ncoinc = 5, 25, 1
    nspec, nwl = flux_2d.shape

    # Run a median filter of the spectra to look for n-sigma outliers.
    # These incantations are kind of complicated but they seem to work
    # i) Compute the median of a wavelength section (nbin) along the observation direction
    # 0,1 : observation,wavelength, row index, column index
    # In moving to numpy v1.10.2, we replaced median with nanmedian
    fmed = np.reshape([np.nanmedian(flux_2d[:, j:j + nbin], axis=1)
                       for j in np.arange(0, nwl, nbin)], (-1, nspec)).T

    # Now expand fmed and flag pixels that are more than outsig off
    fmed_2d = np.reshape([fmed[:, int(j / nbin)] for j in np.arange(nwl)], (-1, nspec)).T

    resid = (flux_2d - fmed_2d) / np.sqrt(fluxvar_2d)
    # If the residual is nan, set flag_2d to 1
    nans = np.isnan(resid)

    flag_2d = np.zeros(nspec * nwl).reshape(nspec, nwl)
    flag_2d[nans] = 1
    flag_2d[~nans] = (np.abs(resid[~nans]) > outsig)

    # If a pixel is flagged in only one spectrum, it's probably a cosmic ray
    # and we should mark it as bad and add ito to badpix_2d.  Otherwise, keep it.
    # This may fail if we coadd many spectra and a cosmic appears in 2 pixels
    # For these cases, we could increase ncoinc
    flagsum = np.tile(np.sum(flag_2d, axis=0), (nspec, 1))
    # flag_2d, flagsum forms a tuple of 2 2d arrays
    # If flag_2d is true and if and flagsum <= ncoinc then set that pixel to bad.
    badpix_2d[np.all([flag_2d, flagsum <= ncoinc], axis=0)] = True


    # Remove bad pixels in the collection of spectra.  In the output they
    # must appear as NaN, but any wavelength bin which is NaN in one spectrum
    # will be NaN in the coadd.  So we need to set the bad pixel values to
    # something innocuous like the median flux, then set the weights of the
    # bad pixels to zero in the coadd.  If a wavelength bin is bad in all
    # the coadds, it's just bad and needs to be marked as NaN in the coadd.
    # In moving to numpy v1.10.2, we replaced median with nanmedian
    flux_2d[badpix_2d] = np.nanmedian(fluxvar_2d)
    fluxvar_2d[badpix_2d] = np.nanmedian(fluxvar_2d)
    badpix_coadd = np.all(badpix_2d, axis=0)
    # Derive the weights
    ## Use just the variance
    wi = 1.0 / (fluxvar_2d)
    # Set the weights of bad data to zero
    wi[badpix_2d] = 0.0
    # Why set the weight of the just first spectrum to np.nan?
    # If just one of the mixels is nan, then the result computed below is nan as well
    for i, val in enumerate(badpix_coadd):
        if val:  wi[0, i] = np.nan

    # Some coadd
    coaddflux = np.average(flux_2d, weights=wi, axis=0)
    coaddfluxvar = np.average(fluxvar_2d, weights=wi, axis=0) / nspec

    coaddflux[badpix_coadd] = np.nan
    coaddfluxvar[badpix_coadd] = np.nan

    # Return the coadded spectrum in a SingleSpectrum object
    return SingleSpec(obj_name, wl, coaddflux, coaddfluxvar, badpix_coadd)


# -------------------------------------------------- #
# ----------------------- BBK ---------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A Brownian Bridge Kernel to use with sklearn       #
# Gaussian Processes to interpolate between          #
# photometry.  I have really just copied             #
# Scikit-learn's RBF kernel and modified it to be a  #
# brownian bridge (sqeuclidian -> euclidian).        #
# -------------------------------------------------- #
class BBK(kernels.StationaryKernelMixin, kernels.NormalizedKernelMixin, kernels.Kernel):
    # Here I am slightly modifying scikit-learn's RBF Kernel to do
    # the brownian bridge.

    """Radial-basis function kernel (aka squared-exponential kernel).
    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length-scale
    parameter length_scale>0, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:
    k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)
    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    .. versionadded:: 0.18
    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return kernels.Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return kernels.Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = kernels._check_length_scale(X, self.length_scale)
        if Y is None:
            # JKH: All I changed was 'sqeuclidean' to 'euclidean'
            dists = pdist(X / length_scale, metric='euclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='euclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])