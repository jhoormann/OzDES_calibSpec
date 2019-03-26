# OzDES_calibSpec
This is a code to perform spectrophotometric calibration.  It was designed to calibrate spectral data from the Anglo Australian Telescope by matching it to near simultaneous photometric observations using DECam on the Blanco Telescope as part of the OzDES Reverberation Mapping Program.

Using the transmission functions for the photometric filters the spectral magnitudes are being matched to, the magnitudes of the spectrum in each band are calculated.  The g,r,i data from DECam span the spectroscopic wavelength range covered by OzDES.  The photometric light curve is linearly interpolated in order to determine the photometric magnitude in each band at the time of the spectroscopic observation.  Scale factors are calculated in each band to convert the spectroscopically derived magnitude into agreement with the calibrated photometric magnitudes.  A 2D polynomial is fit to these scale factors.  This warping function is multiplied by the original spectrum in order to calibrate it.  The uncertainty in this warping function is estimated using Gaussian Processes.  The resulting variance spectrum is combined with the observation variance spectrum supplied for each observation.

The bulk of the calculations is done in the file OzDES_calibSpec_calc.py.  The file OzDES_calibSpec_run.py defines the location and names of all the necessary input and output data and then calls the relevant functions.

# Run Requirements
The code was tested using the following (as stated in requirements.txt)

python==3.5.2

matplotlib==2.0.2

scipy==0.19.1

numpy==1.13.3

astropy==3.0.4

scikit_learn==0.19.0

To run just execute >> python OzDES_calibSpec_run.py

# Input Data
This code will expect you to supply the following data.  The location of this data is defined by the user in OzDES_calibSpec_run.py.
## Spectral Data
This code is currently set up to read in spectral data as outputed by v18 of the OzDES reduction pipeline.  This is a fits file formatted in the following way

Ext 0. Total Coadded Flux

Ext 1. Total Coadded Variance

Ext 2. Total Coadded Bad Pixel Array

Ext 3. Flux for first exposure

Ext 4. Variance for first exposure

Ext 5. Bad Pixel Array for first exposure

Ext 6+. Ext 3-5 repeated for each exposure

The spectral data is read in via the class Spectrumv18.  If your data is stored differently this class can be modified to read in your data and the rest of the code should run without issue.

The units of the flux need to be F-lambda but the scale can be arbitrary.

## Photometric Data

The photometric data is expected to be in a 4 column .txt file with the following labeled columns:

Date    Mag    Mag_err    Band

It is assumed that you have data in 3 photometric filters.  The data are in chronological order by band (ie all g band together followed by r followed by i). For this code to work you need to have a photometric data point in each filter on each side of the spectroscopic observation.
Note while spectroscopic uncertainties are given by variance (sigma^2) photometric are given by sigma.

## Transmission Functions
The transmission function for each of the photometric filters in a two column format: wavelength (nm) and transmission fraction (range 0-1)
These file names and locations are defined in the OzDES_calibSpec_run.py file.

# Output Data
Creates a new fits file with the following information

Ext 0. Scaled flux for first exposure

Ext 1. Scaled variance for first exposure

Ext 2. Bad pixel array for first exposure

Ext 3+. Ext 0-2 repeated for each exposure

Selected information is saved in the header for each extension including the scale factors and photometric magnitudes used in the calibration.

# Reference
If you are using this code please cite the paper where this procedure was first presented,

[Hoormann et al 2019, submitted to MNRAS, arXiv: 1902:04206](https://arxiv.org/abs/1902.04206)
