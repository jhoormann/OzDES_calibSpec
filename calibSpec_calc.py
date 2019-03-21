from astropy.io import fits
import numpy as np
from scipy.integrate import fixed_quad
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import os
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

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



def calibSpec(obj_name, spectra, photo, spectraName, photoName, outBase):
    # Assumes scaling given is of the form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagErr = scaling[9,:]
    # rMag = scaling[10,:]  rMagErr = scaling[11,:]
    # iMag = scaling[12,:]  iMagErr = scaling[13,:]
    #print('numEpochs Before Coadd: ', spectra.numEpochs)

    # Now here is some code that will read in the OzDES data fits files and will
    # organize the individual observations into nights

    listRuns, extensions, badWeather, noPhotometry, badQC = prevent_Excess(obj_name, spectra, photo, bad)

    nevermind, scaling = scaling_Matrix(spectra, extensions, badWeather, noPhotometry, photo)

    extensions = [e for e in extensions if e not in nevermind]

    badWeather = badWeather + nevermind

    coaddFlux = np.zeros((5000, len(listRuns) + 1))
    coaddVar = np.zeros((5000, len(listRuns) + 1))
    coaddBadPix = np.zeros((5000, len(listRuns) + 1))

    speclistC = []  # for coadd - only include spectra for which inCoaddWeather/Photo == True
    index = 1

    #print('numEpochs After Coadd: ', len(listRuns))

    for r in listRuns:
        speclist = []
        for s in extensions:
            if spectra.run[s] == r:
                # scale the spectra
                fluxScale, varScale = warp_spectra(scaling[0:3, s], scaling[3:6, s], spectra.flux[:, s],
                                                   spectra.variance[:, s], spectra.wavelength, r)

    create_output(obj_name, listRuns, coaddFlux, coaddVar, coaddBadPix, extensions, scaling, spectra, redshift, badWeather,
                  noPhotometry, badQC, spectraName, photoName, outBase)
    return

