"""
This script inputs FITS images of subtracted scientific data and:
(i)   locates the corresponding masks, inputs it, and masks the original file;
(ii)  processes the masked image so that its average is zero;
(iii) saves the processed FITS image to a new directory.

The script assumes that the filenames include the name of the field and the
wavelength in which the FITS file has been captured.
"""

import os
import numpy as np
import numpy.ma as ma
from astropy.io import fits
import SZ_funcs as func


path1 = "Data/maps/fbacksub/"  # path to background subtracted images
path2 = "Data/maps/masks/"  # path to masks
daughter = "Data/maps/fbacksub0/"  # daughter directory

fields = np.array(["GAMA9", "GAMA12", "GAMA15", "NGP", "SGP"], dtype=str)  # fields
wavl = np.array(["250", "350", "500"], dtype=str)  # wavelengths
alt_wavl = np.array(["PSW", "PMW", "PLW"], dtype=str)  # alt wavelength names

backsubs = os.listdir(path1)  # list of background subtracted images
masks = os.listdir(path2)  # list of masks

sci_match = func.match(backsubs, fields, wavl, alt_wavl)  # identifies the scientific images
msk_match = func.match(masks, fields, wavl, alt_wavl)  # identifies the masks

for i in range(len(sci_match)):  # loops over backsub matches
	print(backsubs[i])
	for j in range(len(msk_match)):  # loops over mask matches
		if np.array_equal(sci_match[i], msk_match[j]):
			hdulist1 = fits.open(path1 + backsubs[i])
			hdulist2 = fits.open(path2 + masks[j])

			try:  # catches exception for FITS level in some images
				raw = hdulist1[0].data.copy()  # retrieves raw data
			except AttributeError:
				raw = hdulist1[1].data.copy()  # retrieves raw data
			raw[np.isnan(raw)] = 0  # replaces NaN's with zeros
			mask = hdulist2[0].data  # retrieves mask
			hdulist2.close()

			sci = ma.masked_array(raw, np.invert(mask.astype(bool)))  # masks data
			sci -= np.mean(sci)  # setts zero mean
			hdulist1[0].data = ma.filled(sci, 0)  # unmasks data and sets mask=0

			hdulist1.writeto(daughter + "AVGzero_" + backsubs[i])
			hdulist1.close()
