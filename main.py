"""
(1)	Imports the ROSAT catalogue of X-ray sources, and looks for clusters in the
	given Herschel maps. Crops out the clusters and stacks them along an array
	with edge units of normalised cluster radius.
(2)	Performs Monte-Carlo simulations to estimate the noise to the SZ effect.
(3)	Exports the following .npy arrays:
  (i)    the stacked array with the SZ signal 'SZ',              ## shape: (cutsize,cutsize)
  (ii)   the 2d  array of the example MC simulation 'MCsim',     ## shape: (cutsize,cutsize)
  (iii)  the normalised distance array 'distnorm',               ## shape: (nbins)
  (iiv)  the SZ flux array 'fluxring',                           ## shape: (len(wavl),nbins)
  (v)    the corresponding standard deviation error bars 'erb'.  ## shape: (fluxring)
*NOTE	The code also takes into account the flux of the central pixel, although
		it is in a different bin by itself (bin 0). For the central pixel to *not*
		be taken into account, run func.fluxcount(data, nbins, center=False).
(4)	Selects clusters at random and rejects clusters at random to make sure the
	SZ signal does not originate from a small number of clusters.
	'RANDSELECTION' defines 1/RANDSELECTION probability of each cluster to be
	included.
(5)	Computes histogram of fluxes in each annulus (distance bin).
(6)	Fits Gaussian to the histograms of the fluxes in each annulus.
(7)	Calculates and exports mean and stdev, chi-square and p-value of the fit.
(8)	Calculates and exports mean and standard error of sub-mm sources' redshift
	in each distance bin.
"""

import os
import cv2
import numpy as np
import numpy.random as rnd
from scipy.optimize import curve_fit
from scipy.stats import chisquare as chisq
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, Angle
from astropy.cosmology import Planck15 as p15
from astropy.nddata import PartialOverlapError, NoOverlapError
import SZ_funcs as func


## Imports ##
maps = "fbacksub0"
path = "Data/maps/" + maps +"/"  # path to data
savedir = "Data/maps/SZ_effect/" + maps +"/"  # save directory

cats = ["HATLAS_DR1_CATALOGUE_V1.2.DAT",
			"ngp_optical_ids_20170509_sizes.fits",
			"SGP_p1_mf_madx5_01-Dec-2016-042619_noise_14-Mar-2017.dat"]

# loads ROSAT catalogue: RA, DEC, Z, R_500
ROSAT = np.genfromtxt("Data/cats/ROSAT.csv", dtype=str, skip_header=52,
									delimiter="\t", usecols=(2,3,4,7))
c = SkyCoord(ra=ROSAT[:,0], dec=ROSAT[:,1], unit=(u.hourangle,u.deg), frame="icrs")
RA, Dec = c.ra.deg, c.dec.deg  # extracts RA/Dec separately
z = ROSAT[:,2].astype(float)  # redshifts
R_kpc = 1e3*ROSAT[:,3].astype(float)  # physical radii [Mpc]

# loads HATLAS: RA, DEC, RELIABILITY, Z_SPEC
HATLAS = np.genfromtxt("Data/cats/" + cats[0], skip_header=39, usecols=(3,4,32,45))
Gcat = HATLAS[:,:2]  # GAMA (RA/Dec)
# loads NGP catalogue: RA, DEC, RELIABILITY, Z_SPEC
tbdata = fits.open("Data/cats/" + cats[1])[1].data
NGP = map(lambda x: tbdata[x], ["RA", "DEC", "RELIABILITY", "Z_SPEC"])
NGP = np.column_stack(NGP)
Ncat = NGP[:,:2]
# loads SGP catalogue: RA, DEC, RELIABILITY, Z_SPEC
Scat = np.genfromtxt("Data/cats/" + cats[2], skip_header=10, usecols=(3,4))

# stores as coordinate objects
Gc, Nc, Sc = map(lambda x: SkyCoord(ra=x[:,0], dec=x[:,1], unit="deg"), [Gcat, Ncat, Scat])


RANDSELECTION = 1  # 1/RANSELECTION probability of cluster being included  # set to: [1]
cutsize = 51  # edge size of stacking 2d array                             # set to: [51]
pixsize = 3/3600  # minimum pixel size [deg]                               # set to: [1]
MC_repeat = 100  # number of Monte Carlo simulations for each cluster      # set to: [100]
nbins = 7  # number of distance bins                                       # set to: [7]
histbins = 8  # number of flux-histogram-per-bin bins                      # set to: [8]
IRbins = 5  # number of redshift distribution bins                         # set to: [5]
rel_thres = 0.8  # reliability threshold for z_spec                        # set to: [0.8]
over_thres = 0.5  # cutout/map overlap threshold                           # set to: [0.5]
fields = np.array(["GAMA9", "GAMA12", "GAMA15", "NGP", "SGP"])  # fields
wavl = [250, 350, 500]  # wavebands in use [um]
alt_wavl = ["PSW", "PMW", "PLW"]  # alternative waveband naming (short, medium, long)
wavl = map(str, wavl)  # converts to str type
ratios = [469/36, 831/64, 1804/144]  # beam/area ratios at each waveband

naming = {fields[0]: Gcat, fields[1]: Gcat, fields[2]:Gcat,
					fields[3]: Ncat, fields[4]: Scat}  # naming correspondence
metanaming = {fields[0]: HATLAS, fields[1]: HATLAS, fields[2]: HATLAS,
					fields[3]: NGP, fields[4]: None}  # naming correspondence


# SZ and MC simulation arrays
SZ, MCsim = [np.zeros((len(wavl), cutsize, cutsize)) for i in range(2)]
MC = np.zeros((len(wavl), MC_repeat, nbins))  # Monte Carlo array
nstacked = np.zeros(len(wavl))  # number of stacked images at each waveband
rejected = 0  # number of rejected images
z_used = np.array([])  # z of stacked clusters
ncount = np.zeros(nbins)  # sub-mm sources count
# sub-mm sources in clusters' vicinity
sources = [[[],[], [],[], []] for i in range(nbins)]  # distbin, RA, DEC, Z_SPEC, Z
fluxind = [[] for i in range(len(wavl))]

for filename in os.listdir(path):  # loops over all files
	print(filename)
	for pos in range(len(wavl)):  # loops over all wavebands
		if (wavl[pos] in filename) or (alt_wavl[pos] in filename):
			break  # breaks loop when wavelength is in filename

	hdulist = fits.open(path + filename)
	img = hdulist[0].data * ratios[pos]  # image data [Jy/px]

	# Obtains image boundaries in RA/Dec and plots map position on sky
	naxis1 = hdulist[0].header["NAXIS1"]  # image width
	naxis2 = hdulist[0].header["NAXIS2"]  # image height

	w = WCS(hdulist[0].header)  # obtains WCS information

	RA_max, DEC_min = w.wcs_pix2world([[1,1]], 1)[0]  # bottom-left edge (SE)
	RA_min, DEC_max = w.wcs_pix2world([[naxis1, naxis2]], 1)[0]  # top-right edge (NW)

	# finds indices of the clusters, the centers of which are inside the image field
	boundaries = (RA_min, RA_max, DEC_min, DEC_max)  # image boundaries
	infield = func.is_infield(RA, Dec, boundaries)

	if len(infield[infield==True]) != 0:
		# cosmological distance calculator -- angular radius [arcsec]
		R_map = R_kpc[infield]*p15.arcsec_per_kpc_comoving(z[infield])*u.kpc
	else:  # continues if no clusters are found
		continue

	# rebins the image with 'pixscale' and updates header
	# assumes square pixels
	hdr = hdulist[0].header.copy()  # copies original header
	old_pixsize = np.abs(hdulist[0].header["CDELT1"])
	zoom = old_pixsize/pixsize  # computes image scale factor

	rebin = cv2.resize(img.astype(float), dsize=(0,0),
					fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)  # rebins image
	rebin /= zoom**2  # normalised image data [Jy/px]

	hdr["NAXIS2"], hdr["NAXIS1"] = np.array(np.shape(rebin))  # dimensions
	hdr["CDELT2"], hdr["CDELT1"] = pixsize, -pixsize  # pix size
	hdr["CRPIX2"], hdr["CRPIX1"] = np.array([
									hdulist[0].header["CRPIX2"]*zoom,
									hdulist[0].header["CRPIX1"]*zoom])  # ref pixels
	w_new = WCS(hdr)  # new WCS


	for i, R in enumerate(R_map):  # loops over clusters in field
		## Cluster Selection ##
		ra_c, dec_c = RA[infield][i], Dec[infield][i]  # cluster central coords
		coords = SkyCoord(ra_c, dec_c, unit="deg")  # converts to SkyCoord object

		try:  # rejects if cluster is not fully enclosed
			box = Cutout2D(rebin, coords, 2*R, wcs=w_new, mode="strict")
			# 'strict' mode raises exception if overlap is not 100%
			cut = box.data  # image data of cutout
			if np.count_nonzero(cut) == 0:  # rejects fully masked clusters
				rejected += 1  # rejected cluster count
				continue
		except (PartialOverlapError, NoOverlapError, ValueError):
			rejected += 1  # rejected cluster count
			continue

		# resizes cutout to 'cutsize' using bicubic interpolation
		cut = cv2.resize(cut, dsize=(cutsize,cutsize), fx=0, fy=0,
						interpolation=cv2.INTER_CUBIC)

		overlap = len(cut[cut!=0])/cutsize**2  # fractional overlap
		if overlap < over_thres: continue
		# Selects cluster at random
		k = rnd.randint(RANDSELECTION)
		if k != 0: continue  # continues if random is not 0

		## Cutout Manipulation ##
		unzoom = float(len(cut))/cutsize  # image resize scale factor
		cut *= unzoom**2  # normalises cutout [Jy/px]
		cut = func.clipping(cut)

		## Cluster Redshift ##
		if pos == 0: z_used = np.append(z_used, z[infield][i])

		## Individual Flux Count ##
		p = func.fluxcount(cut, nbins, center=False)
		fluxind[pos].append(p.fluxes())

		nstacked[pos] += 1  # stacked images count
		SZ[pos] += cut  # adds flux data to SZ stack array

		## sub-mm Sources ##
		if pos == 0:  # go over all sub-mm sources only once
			R_deg = R.value/3600  # cluster radius [deg]
			bounds = (ra_c-R_deg, ra_c+R_deg, dec_c-R_deg, dec_c+R_deg)  # box boundaries
			center = SkyCoord(ra=ra_c*u.deg, dec=dec_c*u.deg)  # center coords

			for k in range(len(fields)):
				if (fields[k] or fields[k].swapcase()) in filename:
					try:
						cat = naming[fields[k]]  # sub-mm catalogue
						metacat = metanaming[fields[k]]  # full catalogue
						inbox = func.is_infield(cat[:,0], cat[:,1], bounds)
						cat = cat[inbox]
						metacat = metacat[inbox]
					except TypeError: pass  # if catalogue does not exist
				else: continue

				catcoords = SkyCoord(ra=cat[:,0]*u.deg, dec=cat[:,1]*u.deg)
				dist = np.array([source.separation(center).deg
										for source in catcoords])  # distances

				for m, distance in enumerate(dist):
					# cutout is square; distance is inscribed circle
					if distance <= R_deg:
						distbin = int((distance*nbins)//R_deg)
						ncount[distbin] += 1

						if metacat is not None:
							# Appends metadata
							sources[distbin][0].append(distbin+1)
							sources[distbin][1].append(cat[m,0])
							sources[distbin][2].append(cat[m,1])
							sources[distbin][4].append(z[infield][i])
							# appends redshift if reliability > threshold
							if metacat[m,2] >= rel_thres:
								sources[distbin][3].append(metacat[m,3])
							else: sources[distbin][3].append(-1)

				break

#"""
		## Random Sampling ##
		R_map_deg = Angle(R_map[i]).deg  # converts to deg
		extract = rnd.randint(MC_repeat)  # random simulation extraction
		for sim in range(MC_repeat):  # loops MC_repeat times for the MC simulations
			while True:
				# random RA/Dec for random sampling (RA accounts for PBCs in ICRS frame)
				ra_rand =  rnd.uniform(RA_min + R_map_deg, RA_max + 360 - R_map_deg)
				dec_rand = rnd.uniform(DEC_min + R_map_deg, DEC_max - R_map_deg)
				coords_rand = SkyCoord(ra_rand % 360, dec_rand, unit="deg")
				# cuts random sample
				try:  # catches exception if cluster is at edge
					boxrand = Cutout2D(rebin, coords_rand, 2*R_map[i], wcs=w_new, mode="strict")
					cutrand = boxrand.data  # image data of random cutout
				except (PartialOverlapError, NoOverlapError, ValueError):
					continue

				if np.count_nonzero(cutrand) == len(cutrand)**2:
					break  # breaks if cutrand is not masked

			cutrand = cv2.resize(cutrand, dsize=(cutsize, cutsize), fx=0, fy=0,
								interpolation=cv2.INTER_CUBIC)  # resizes 'cutrand'

			q = func.fluxcount(cutrand, nbins, center=False)
			error = q.fluxes()  # computes mean fluctuation
			MC[pos,sim] += error*2  # adding the variance

			if sim == extract:
				MCsim[pos] += cutrand  # adds random simulation
"""
#"""
# Manipulating 'sources' array to make it exportable
sources = np.array(sources)
sources = [np.column_stack(sources[i]) for i in range(nbins)]
sources = np.vstack(sources)
# Statistics of sub-mm sources
redshifts = sources[:,3][sources[:,3]>0]  # extracts sources with redshifts
whichbin = sources[:,0][sources[:,3]>0]  # extracts bins for sources with redshifts
clustred = sources[:,4][sources[:,3]>0]  # extracts cluster redshifts

n1, n2, z_mean, z_std, zr_mean, zr_std = [np.array([]) for i in range(6)]
for i in range(1, nbins+1):
	n1 = np.append(n1, len(sources[sources[:,0]==i]))
	n2 = np.append(n2, len(sources[(sources[:,0]==i) & (sources[:,3]>0)]))

	z_mean = np.append(z_mean, redshifts[whichbin==i].mean())
	z_std = np.append(z_std, redshifts[whichbin==i].std() / len(redshifts[whichbin==i]))  # standard error
	zr_mean = np.append(zr_mean, np.mean(redshifts[whichbin==i] / clustred[whichbin==i]))
	zr_std = np.append(zr_std, np.std(redshifts[whichbin==i] / clustred[whichbin==i])
						/ len(clustred[whichbin==i]))  # standard error

IRratio = n2/n1  # fraction of sources of known redshift in each bin


# averages signal in SZ and MCsim arrays
# calculates flux in each bin and in each wavelength
# calculates error bars in each bin and in each wavelength
# calculates the distribution of fluxes inside each bin for each wavelength
# returns the probability density of the fluxes and the midpoint of each bin
# calculates the best-fitting Gaussian for each distance bin
# returns the optimal mean (mu) and sigma (std) of the Gaussian fit
# returns the chi-squared value of the fit and the corresponding p-value
fluxring, erb = [np.zeros((len(wavl), nbins)) for i in range(2)]
fluxhist, fluxbins = [np.zeros((len(wavl), nbins, histbins)) for i in range(2)]
popt = np.zeros((len(wavl), nbins, 3))
pstat = np.zeros((len(wavl), nbins, 3))
q2 = np.zeros((len(wavl), nbins, histbins))
chi2, pval = [np.zeros((len(wavl), nbins)) for i in range(2)]
for i in range(len(wavl)):
	SZ[i] /= nstacked[i]
	MCsim[i] /= nstacked[i]
	MC[i] /= nstacked[i]

	q = func.fluxcount(SZ[i], nbins, center=False)
	fluxring[i] = q.fluxes()  # computes the flux in each distance bin
	fluxhist[i], fluxbins[i] = q.histplot(histbins, density=False, midpoint=True)
	for j in range(nbins):  # loops over all distance bins
		erb[i,j] = MC[i,:,j].std()

		# Fits Gaussian curve to histograms
		xdata, ydata = fluxbins[i][j], fluxhist[i][j]
		size = sum(ydata)
		pstat[i,j] = [sum(ydata), fluxring[i,j], erb[i,j]]  # guess parameters
		popt[i,j], _ = curve_fit(func.Gauss, xdata, ydata, p0=pstat[i,j])

		# Gaussian fitting
		q1 = np.linspace(xdata.min(), xdata.max(), len(xdata))
		q2[i,j] = func.Gauss(q1, *popt[i,j])

		chi2[i,j], pval[i,j] = chisq(ydata, q2[i,j], ddof=3)

distnorm = q.distnorm  # distance array
Anorm = np.pi*distnorm**2 - np.pi*(distnorm - 1/nbins)**2  # area normalisation
#"""
#"""
## Exporting ##
popt, pstat = map(lambda x: x[:,:,1:], [popt, pstat])  # mean & std of fit & stat
hdr = "distbin pstat_mu pstat_std popt_mu popt_std chi_sq pval_3dof"  # columns
histbinnumber = np.arange(1, nbins+1, 1)
# Exports one file for each wavelength with the
for i in range(len(wavl)):
	fname = "gaussfit%s.dat" % wavl[i]
	X = np.column_stack((histbinnumber, pstat[i], popt[i], chi2[i], pval[i]))
	np.savetxt(savedir + fname, X, fmt="%.10e", header=hdr, comments="")

hdr = "DISTBIN RA DEC Z_SPEC"
fname = "submm_sources.dat"
np.savetxt(savedir + fname, sources, header=hdr, comments="")

np.save(savedir + "SZ", SZ)
np.save(savedir + "MCsim", MCsim)
np.save(savedir + "distnorm", distnorm)
np.save(savedir + "fluxring", fluxring)
np.save(savedir + "erb", erb)
np.save(savedir + "fluxhist", fluxhist)
np.save(savedir + "fluxbins", fluxbins)
np.save(savedir + "z", z)
np.save(savedir + "z_used", z_used)
np.save(savedir + "ncount", ncount)
np.save(savedir + "Anorm", Anorm)
np.save(savedir + "fluxind", fluxind)
np.save(savedir + "IRratio", IRratio)
np.save(savedir + "z_mean", z_mean)
np.save(savedir + "z_std", z_std)
np.save(savedir + "zr_mean", zr_mean)
np.save(savedir + "zr_std", zr_std)
#"""
