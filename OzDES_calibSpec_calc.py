# ---------------------------------------------------------- #
# This is a code to perform spectrophotometric calibration.  #
# It was designed to calibrate spectral data from the Anglo  #
# Australian Telescope by matching it to near simultaneous   #
# photometric observations using DECam on the Blanco         #
# Telescope as part of the OzDES Reverberation Mapping       #
# Program.  Unless otherwise noted this code was written by  #
# Janie Hoormann.                                            #
# ---------------------------------------------------------- #

from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import matplotlib.pyplot as plt


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

def calibSpec(obj_name, spectra, photo, spectraName, photoName, outBase, bands, filters, centers, plotFlag):
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
    extensions, noPhotometry, badQC = prevent_Excess(spectra, photo, bands)

    # Then we calculate the scale factors
    nevermind, scaling = scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, bands, filters)

    # Remove last minute trouble makers
    extensions = [e for e in extensions if e not in nevermind]
    badQC = badQC + nevermind

    # And finally warp the data
    for s in extensions:
        # scale the spectra
        spectra.flux[:, s], spectra.variance[:, s] = warp_spectra(scaling[0:3, s], scaling[3:6, s], spectra.flux[:, s],
                                                                  spectra.variance[:, s], spectra.wavelength, centers,
                                                                  plotFlag)

    create_output(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase)

    return

# -------------------------------------------------- #
# ---------------- prevent_Excess ------------------ #
# -------------------------------------------------- #
# This function removes extensions from the list to  #
# calibrate because of insufficient photometric data #
# or bad quality flags                               #
# -------------------------------------------------- #

def prevent_Excess(spectra, photo, bands):

    # First, find the min/max date for which we have photometry taken on each side of the spectroscopic observation
    # This will be done by finding the highest date for which we have photometry in each band
    # and taking the max/min of those values
    # This is done because we perform a linear interpolation between photometric data points to estimate the magnitudes
    # observed at the specific time of the spectroscopic observation

    maxPhot = np.zeros(3)

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
    photLim = min(maxPhot)

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
    photLimMin = max(minPhot)
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

def scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, bands, filters):
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
# originally written by Dale Mudd                    #
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
# time of the spectral observation                   #
# -------------------------------------------------- #

def des_photo(photo, spectral_mjd, bands):

    """Takes in an mjd from the spectra, looks through a light curve file to find the nearest photometric epochs and
    performs linear interpolation to get estimate at date, return the photo mags."""

    # !!! Assumes dates are in chronological order

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

    g_mag, g_mag_err = interpolatePhot(g_date_v, g_mag_v, g_err_v, spectral_mjd)
    r_mag, r_mag_err = interpolatePhot(r_date_v, r_mag_v, r_err_v, spectral_mjd)
    i_mag, i_mag_err = interpolatePhot(i_date_v, i_mag_v, i_err_v, spectral_mjd)

    return [g_mag, r_mag, i_mag], [g_mag_err, r_mag_err, i_mag_err]

# -------------------------------------------------- #
# --------------- interpolatePhot  ----------------- #
# -------------------------------------------------- #
# Performs linear interpolation and propagates the    #
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
    scale_factor_sigma = mag_diff_var * (scale_factor * 0.4 * 2.3) ** 2

    return scale_factor, scale_factor_sigma

# -------------------------------------------------- #
# ----------------- warp_spectra  ------------------ #
# -------------------------------------------------- #
# Fits polynomial to scale factors and estimates     #
# associated uncertainties with gaussian processes.  #
# -------------------------------------------------- #

def warp_spectra(scaling, scaleErr, flux, variance, wavelength, centers, plotFlag):

    # associate scale factors with centers of bands and fit 2D polynomial to form scale function.
    scale = InterpolatedUnivariateSpline(centers, scaling, k=2)
    fluxScale = flux * scale(wavelength)

    # add in Gaussian process to estimate uncertainties, /10**-17 because it gets a bit panicky if you use small numbers
    stddev = (scaleErr ** 0.5) / 10 ** -17
    scale_v = scaling / 10 ** -17

    kernel = kernels.RBF(length_scale=300, length_scale_bounds=(0.1, 2000.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=stddev)

    xprime = np.atleast_2d(centers).T
    yprime = np.atleast_2d(scale_v).T

    gp.fit(xprime, yprime)
    xplot_prime = np.atleast_2d(wavelength).T
    y_pred, sigma = gp.predict(xplot_prime, return_std=True)

    sigma = sigma * 10 ** -17

    # now scale the original variance and combine with scale factor uncertainty
    varScale = variance * pow(scale(wavelength), 2) + sigma ** 2

    '''
    fig, (ax2, ax3) = plt.subplots(2, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(15, 20, forward=True)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(25)
    fig.subplots_adjust(hspace=0)

    ax2.set_ylabel("f$_\lambda$", **axis_font)
    ax2.set_xlabel("Wavelength ($\AA$)", **axis_font)
    ax2.set_title(str(r), **title_font)
    ax2.plot(wavelength, flux, color='black', label="Before Scaling")
    ax2.legend(loc=1, frameon=False, prop={'size': 20})
    ax3.set_ylabel("f$_\lambda$  (10$^{-17}$ erg/s/cm$^2$/$\AA$)", **axis_font)
    ax3.set_xlabel("Wavelength ($\AA$)", **axis_font)
    ax3.plot(wavelength, fluxScale / 10 ** -17, color='black', label="After Scaling")
    ax3.legend(loc=1, frameon=False, prop={'size': 20})

    plt.show()
    fig.clear()
    '''

    return fluxScale, varScale

# -------------------------------------------------- #
# ---------------- create_output  ------------------ #
# -------------------------------------------------- #
# Outputs the warped spectra to a new fits file.     #
# -------------------------------------------------- #


def create_output(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase):

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
        header['CRPIX1'] = spectra.crpix1
        header['CRVAL1'] = spectra.crval1
        header['CDELT1'] = spectra.cdelt1
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['EPOCHS'] = len(extensions)

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
# Modified from code originally provided by          #
# Harry Hobson                                       #
# -------------------------------------------------- #
# ------------------ mark_as_bad ------------------- #
# -------------------------------------------------- #
# Occasionally you get some big spikes in the data   #
# that you do not want messing with your magnitude   #
# calculations.  Remove these by looking at single   #
# bins that have a significantly larger than average #
# fluxes or variances and change those to nans.      #
# Nans will be interpolated over.                    #
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

            if np.isnan(variance[i]) == False and variance[i] > 3.5*avg:

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

        flux[0] = 0.0
        flux[-1] = 0.0
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

