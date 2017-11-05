import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import iqr
from scipy.stats import norm as normal


class fluxcount:
	""" Handles operations on the counting of the flux within circular annuli
	around the central pixel of the 2d ``data`` array.
	"""

	def __init__(self, data, nbins, center=False):
		self.data = data
		self.nbins = nbins
		self.size = len(data)
		self.center = center

		self.distbins = np.linspace(0, self.size/2., nbins+1)  # distance bins
		distarray = self.dist()  # distances
		self.distmask = np.digitize(distarray, self.distbins, right=True)  # digitizes distances


	def dist(self):
		""" Calculates the euclidean distance between a pixel and the center
		of an image.
		"""
		cpix = np.array([np.ceil(self.size/2.), np.ceil(self.size/2.)])  # center coords

		distarray = np.zeros((self.size, self.size))  # creates a square 2d array
		for i in range(self.size):
			for j in range(self.size):
				w = np.array([i,j])  # position of pixel
				distarray[i,j] = np.linalg.norm(w - cpix)  # Euclidean distance

		return distarray


	def fluxes(self):
		""" Returns an array of size ``nbins`` with the mean flux inside each
		distance bin (annulus).
		"""
		self.distnorm = self.distbins[1:]/self.distbins[-1]  # normalised distances

		F = np.array([np.mean(self.data[self.distmask == i+1])
								for i in range(self.nbins)])

		if self.center:  # accounts for the central pixel
			med = int(np.median(range(1, self.size+1)))  # median pixel number
			F[0] += self.data[med, med]

		return F


	def stdev(self):
		""" Returns an array of size ``nbins`` with the standard deviation of
		the flux inside each distance bin (annulus).
		"""
		S = np.array([np.std(self.data[self.distmask == i+1])
								for i in range(self.nbins)])

		return S


	def histplot(self, histbins, density=True, midpoint=False):
		hist, bins = [[[]]*self.nbins for i in range(2)]
		for i in range(self.nbins):
			hist[i], bins[i] = np.histogram(self.data[self.distmask == i+1],
											bins=histbins, density=density)

		if midpoint:  # returns the midpoint of the bin edges
			bins = [(bins[i][1:] + bins[i][:-1])/2 for i in range(self.nbins)]

		return hist, bins


def match(files, fields, wavls, alt_wavls=None):
	"""	Returns a matching array. Each row gives the indices of the field and
	wavelength of each image.
	"""
	matching = np.zeros((len(files), 2))  # matching 2d array

	for i in range(len(files)):  # iterates over all backsub filenames
		for j in range(len(fields)):  # loops over all fields
			if (fields[j] or fields[j].swapcase()) in files[i]:  # identifies the field
				matching[i,0] = j
				break
		for k in range(len(wavls)):  # loops over all wavelengths
			if wavls[k] in files[i]:  # identifies the wavelength
				matching[i,1] = k
				break
			if alt_wavls is not None:  # checks if alt wavelengths are given
				if alt_wavls[k] in files[i]:  # identifies alt wavelength
					matching[i,1] = k
					break

	return matching


def is_infield(RA, Dec, img_bounds):
	""" Returns clusters in the field.
	"""
	ra_min, ra_max, dec_min, dec_max = img_bounds
	if ra_min < ra_max:
		infield = (RA > ra_min) & (RA < ra_max) & (Dec > dec_min) & (Dec < dec_max)
	else:
		infield = ((RA > ra_min) | (RA < ra_max)) & (Dec > dec_min) & (Dec < dec_max)

	return infield


def Gauss(x, A, mu, sigma):
	""" Computes a gaussian distribution of x.
	"""
	p = A * np.exp(-(x-mu)**2/(2*sigma**2))
	return p


def clipping(cut, sigma_thres=5):
	""" Masks array elements with extreme values.
	"""
	iqr2std = 1/normal.ppf(0.75)  # IQR to SD

	x0 = np.median(cut)  # mean estimate
	std = iqr(cut)*iqr2std  # stdev estimate

	mask = cut > x0 + sigma_thres*std  # flags extreme values
	clipped = MaskedArray(cut, mask)

	return clipped
