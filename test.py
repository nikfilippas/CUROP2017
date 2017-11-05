"""
Creates a test image for the SZ effect.
"""

import numpy as np
import numpy.random as rnd
import numpy.ma as ma
import SZ_funcs as func

cutsize = 51
nbins = 7

img = nbins*rnd.rand(cutsize, cutsize)
q1 = func.fluxcount(img, nbins)
distarray = q1.dist()
distbins = q1.distbins
mask = distarray >= distbins[-1]
img = ma.masked_array(img, mask=mask)

FLUX = np.zeros(nbins)
for i in range(nbins):
	isinbin = (distarray >= distbins[i]) & (distarray < distbins[i+1])

	img[isinbin] /= i+1
	FLUX[i] = img[isinbin].mean()

q2 = func.fluxcount(img, nbins)
fluxes = q2.fluxes()

print(FLUX)
print(fluxes)
