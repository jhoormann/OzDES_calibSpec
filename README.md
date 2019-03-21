# calibSpec
This is a code to perform spectrophotometric calibration.  It was designed to calibrate spectral data from the Anglo Australian Telescope by matching it to near simultaneous photometric observations using DECam on the Blanco Telescope as part of the OzDES Reverberation Mapping Program.


# Run Requirements


# Input Data
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


## Photometric Data

The photometric data is expected to be in a 4 column .txt file with the following labeled columns:

MJD   Mag   Mag_err   Band


## Transmission Functions


# Output Data

# Reference
If you are using this code please cite the paper where this procedure was first presented,

[Hoormann et al 2019, submitted to MNRAS, arXiv: 1902:04206](https://arxiv.org/abs/1902.04206)
