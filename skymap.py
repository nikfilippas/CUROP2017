"""
Plots skymap of position of the Herschel fields, taking masked areas into account.
"""

from __future__ import division
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


path = "Data/maps/masks/"  # path to data
ROSAT = np.genfromtxt("Data/cats/ROSAT.csv", dtype=str, skip_header=52,
					delimiter="\t", usecols=(2,3,4,7))  # loads ROSAT
c = SkyCoord(ra=ROSAT[:,0], dec=ROSAT[:,1], unit=(u.hour,u.deg), frame="icrs")
wavl = [250, 350, 500]  # waveband in use [um]
alt_wavl = ["PSW", "PMW", "PLW"]  # alternative waveband naming (short, medium, long)
wavl = map(str, wavl)  # converts to str type

# ROSAT map with Herschel map cover
ra, dec = c.ra.wrap_at(180*u.deg).rad, c.dec.wrap_at(180*u.deg).rad  # extracts RA/Dec of X-ray sources
# pinpoints NGP, SGP, and Equator
NGP, SGP = map(lambda x: SkyCoord(l=0*u.deg, b=x*u.deg, frame="galactic"), [90,-90])

l_gal = np.linspace(-180, 179.99, 1000)  # galactic plane longitude
ra_gal = SkyCoord(l=l_gal*u.deg, b=0*u.deg, frame="galactic").icrs.ra.value  # extracts RA
gal0_index = np.argmax(ra_gal) - 499  # index of point where RA=0 (-499 due to (-180,180) range)
l_gal = np.append(l_gal[gal0_index:], l_gal[:gal0_index])  # galactic plane in icrs
c_gal = SkyCoord(l=l_gal*u.deg, b=0*u.deg, frame="galactic")  # stores as coord object

NGP, SGP, galplane = map(lambda x: x.icrs, [NGP, SGP, c_gal])
# Plots the points
ax = plt.subplot(111, projection="mollweide")
## modifies tick labels
h = np.arange(14, 14+24, 2)[:-1]
h = map(lambda x: str(x%24)+"h", h)
ax.set_xticklabels(h)
ax.grid("on", which="both", ls=":", c="grey")

map(lambda x: ax.plot(x.ra.wrap_at(180*u.deg).rad, x.dec.rad, "k+", ms=10, mew=3), [NGP,SGP])
ax.plot(galplane.ra.wrap_at(180*u.deg).rad, galplane.dec.rad, "k--", lw=2)
ax.plot(ra, dec, "rx", ms=1)

for filename in os.listdir(path):  # loops over all files
	for pos in range(len(wavl)):  # loops over all wavebands
		if (wavl[pos] in filename) or (alt_wavl[pos] in filename):
			break  # breaks loop when wavelength is in filename

	if pos == 2:  # plots map position on sky only once (no overlaps)
		print(filename)
		hdulist = fits.open(path + filename)
		# resizes data to preserve memory
		data = cv2.resize(hdulist[0].data, dsize=(0,0), fx=0.1, fy=0.1,
										interpolation=cv2.INTER_NEAREST)
		data = data/data.max()  # normalising the data

		# Obtains image boundaries in RA/Dec and plots map position on sky
		naxis1 = hdulist[0].header["NAXIS1"]  # image width
		naxis2 = hdulist[0].header["NAXIS2"]  # image height

		w = WCS(hdulist[0].header)  # obtains WCS information

		ra_max, dec_min = w.wcs_pix2world([[1,1]], 1)[0]  # bottom-left edge (SE)
		ra_min, dec_max = w.wcs_pix2world([[naxis1, naxis2]], 1)[0]  # top-right edge (NW)
		# Converts to rad
		ra_min, ra_max = map(lambda x: np.deg2rad(x), [ra_min, ra_max])
		dec_min, dec_max = map(lambda x: np.deg2rad(x), [dec_min, dec_max])
		# Accounts for PBCs
		_wraps = False
		if ra_min > ra_max:
			ra_max += 2*np.pi
		if (ra_min > np.pi) and (ra_max > np.pi):
			ra_min -= 2*np.pi
			ra_max -= 2*np.pi
		if (ra_max > np.pi) and (ra_min < np.pi):
			ra_max -= 2*np.pi
			_wraps = True

		if not _wraps:
			extent = (ra_min, ra_max, dec_min, dec_max)  # extent of image
			ax.imshow(data, cmap="Greys", vmin=0, vmax=2, extent=extent)
		else:  # accounts for PBCs
			extent1 = (ra_min, np.pi, dec_min, dec_max)  # extent 1 of image
			extent2 = (-np.pi, ra_max, dec_min, dec_max)  # extent 2 of image
			ax.imshow(data, cmap="Greys", vmin=0, vmax=2, extent=extent1)
			ax.imshow(data, cmap="Greys", vmin=0, vmax=2, extent=extent2)

ax.set_aspect(0.5)
#plt.savefig("Data/maps/skymap_new.png", dpi=1200, bbox_inches="tight")
#plt.close()
