"""
This script takes .npy inputs of the following 2d arrays:
(i)    the stacked array with the SZ signal,
(ii)   the array with the random stacking,
and plots graphs of the signal received within an array of radii in two ways:
(i)    the signal received within each bin separately,
(ii)   the cumulative signal received up to each bin's radius.
"""

from __future__ import division
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from astropy.visualization import ImageNormalize, PowerStretch
from astropy.visualization import ZScaleInterval as zscale


maps = "fbacksub0"
path = "Data/maps/SZ_effect/" + maps + "/"  # path to data
savedir = "Data/maps/images/" + maps + "/"  # save directory

pwr = 1e+3  # power of 10 used to plot data (axes labels change)
SZ = np.load(path + "SZ.npy")*pwr  # SZ signal
MCsim = np.load(path + "MCsim.npy")*pwr  # Monte Carlo simulation example
distnorm = np.load(path + "distnorm.npy")  # normalised distances
fluxring = np.load(path + "fluxring.npy")*pwr  # SZ fluxes
erb = np.load(path + "erb.npy")*pwr  # error bars of SZ fluxes
fluxbins = np.load(path + "fluxbins.npy")*pwr  # bins for bin fluxes
fluxhist = np.load(path + "fluxhist.npy")  # fluxes in each bin
fluxind = np.load(path + "fluxind.npy")*pwr  # individual SZ fluxes
z = np.load(path + "z.npy")  # ROSAT z
z_used = np.load(path + "z_used.npy")  # used z from ROSAT
ncount = np.load(path + "ncount.npy")  # source count in each distance bin
Anorm = np.load(path + "Anorm.npy")  # normalisation of the distance bins
IRratio = np.load(path + "IRratio.npy")  # fraction of sources with measured redshift
z_mean = np.load(path + "z_mean.npy")  # mean redshift in each distance bin
z_std = np.load(path + "z_std.npy")  # stdev of redshift in each distance bin
zr_mean = np.load(path + "zr_mean.npy")  # mean redshift ratio of sub-mm to cluster
zr_std = np.load(path + "zr_std.npy")  # std of the redshift ratio of sub-mm to cluster

cutsize = len(SZ)  # edge length of images
nbins = len(fluxbins[0])  # number of distance bins
histbins = len(fluxbins[0][0])  # number of histogram bins
med = int(np.median(range(1, cutsize+1)))  # median pixel number
wavl = [250, 350, 500]  # waveband in use [um]
wavl = map(str, wavl)  # converts to str type
distance = distnorm - (distnorm[0]/2)  # radius at bin midpoint


#"""
## Plotting ##
# Figure 1 #
fig1, ax1 = plt.subplots(len(wavl), 1, sharex=True, figsize=(6,3*len(wavl)))
[ax1[i].grid("on", which="both", ls=":") for i in range(len(ax1))]
ax1[-1].set_xlabel(r"$r \mathrm{ / R_{500}}$", fontsize=16)
mid = int(np.median(range(len(wavl))))  # central plot
ax1[mid].set_ylabel(r"$\mathrm{F_{\lambda} \/\/ \left[ mJy/px \right]}$", fontsize=16)
null = np.append(np.zeros(1), distnorm)  # reference line

# Figure 2 #
fig2, ax2 = plt.subplots(3, len(wavl), figsize=(3*len(wavl),6.5),
						 gridspec_kw={"height_ratios":[0.05,1,1]})
cax, ax2 = ax2[0], ax2[1:]  # distinguishes axes, colorbars

for i in range(len(wavl)):
	wav = "$\\mathrm{ %s \/ \\mu m}$" % wavl[i]
	ctr = SZ[med,med,i]  # central pixel value
	ax1[i].text(0.83, 0.90, wav, fontsize=14, backgroundcolor="white",
										transform=ax1[i].transAxes)

	ax1[i].plot(null, np.zeros_like(null), "r:", lw=2)
	ax1[i].errorbar(distance, fluxring[i], yerr=erb[i],
					fmt="ko-", lw=3, elinewidth=1, capsize=2, label=wav)

	# Plots SZ signal
	# normalises data with 'zscale' and stretches it by 'PowerStretch' as in ds9
	norm = ImageNormalize(SZ[i], interval=zscale(), stretch=PowerStretch(1.3))
	P1 = ax2[1,i].imshow(MCsim[i], norm=norm, aspect="auto")
	P2 = ax2[0,i].imshow(SZ[i], norm=norm, aspect="auto")
	cb = fig2.colorbar(P1, cax=cax[i], orientation="horizontal")
	cb.ax.xaxis.set_ticks_position("top")
	cb.set_label(r"$\mathrm{mJy/px}$", fontsize=10)
	cb.ax.xaxis.set_label_position("top")
	cb.ax.tick_params(labelsize=8)


	for j in range(len(ax2)):
		ax2[j,i].axis("off")

		t = ax2[j,i].text(0.69,0.90, wav, color="w", transform=ax2[j,i].transAxes,
															fontsize=14)
		t.set_bbox(dict(facecolor="k", alpha=0.4))

fig1.tight_layout()
fig1.savefig(savedir + "SZ.png", dpi=1200, bbox_inches="tight")
fig2.tight_layout(w_pad=0, h_pad=0)
fig2.savefig(savedir + "SZ_pix.png", dpi=1200, bbox_inches="tight")


# Figure 3 #
cm_subsection = np.linspace(0, 1, histbins)
colors = [cm.plasma(x) for x in cm_subsection]

fig3, ax3 = plt.subplots(1, len(wavl), figsize=(3*len(wavl), 3), sharey=True)
fig3.text(0.5, -0.03, r"$\mathrm{F_{\lambda} / mJy}$", ha="center", fontsize=16)
fig3.text(-0.01, 0.5, r"$\mathrm{N}$", fontsize=16, ha="center", rotation="vertical")

# sets distances as d_n where n is the bin number
lines = ["$\\mathrm{%.2f}$" % distance[i] for i in range(nbins)]
for i in range(len(wavl)):
	wav = "$\\mathrm{ %s \/ \\mu m}$" % wavl[i]
	ax3[i].grid("on", which="both", ls=":")
	ax3[i].text(0.5, 1, wav, fontsize=14, ha="center", va="bottom",
									transform=ax3[i].transAxes)
	ax3[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


	# Indicates the mean and std of the first distance bin on the plot
	mn, sd = fluxring[0,i], erb[0,i]

	x = np.array([mn-sd, mn+sd])  # x bounds
	y = fluxhist.max() * np.ones_like(x)  # y bound
	ax3[i].fill_between(x, y, color="grey", alpha=0.4)

	x0 = np.array([mn, mn])  # centerline (x)
	y0 = np.array([0, fluxhist.max()])  # centerline (y)
	ax3[i].plot(x0, y0, "k-.", lw=2, alpha=0.4)

	for j in range(nbins):
		xdata, ydata = fluxbins[i][j], fluxhist[i][j]
		size = sum(ydata)
		line, = ax3[i].plot(xdata, ydata, "o-", ms=4, c=colors[j])
		if i == len(wavl)-1: line.set_label(lines[j])  # custom label for legend


ax3[0].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))  # integer yaxis
box = (1,0,1,1)
lgd = ax3[-1].legend(loc="center", frameon=False, fontsize=12,
					bbox_to_anchor=(1.2,0.5), bbox_transform=ax3[-1].transAxes)
lgd.set_title(r"$\bar r \mathrm{ / R_{500}}$", prop={"size":"x-large"})

fig3.tight_layout()
fig3.savefig(savedir + "binfluxes.png", dpi=1200, bbox_inches="tight")


# Figure 4 #
fig4, ax4 = plt.subplots(1,1)
ax4.set_yscale("log")
ax4.set_xlabel("$\mathrm{z}$", fontsize=16)
ax4.set_ylabel("$\mathrm{N}$", fontsize=16)

hist4, bins4, _ = ax4.hist(z, bins=15, histtype="step", color="k", ls=":", lw=3,
						label="$\\mathrm{ROSAT \/ (%d \/ clusters)}$" % len(z))
ax4.hist(z_used, bins=bins4, histtype="step", color="k", ls="-", lw=3,
						label="$\\mathrm{used \/ (%d \/ clusters)}$" % len(z_used))

ax4.legend(loc="upper right", fontsize=14, fancybox=True)
fig4.tight_layout()
fig4.savefig(savedir + "z_hist.png", dpi=1200, bbox_inches="tight")


# Figure 5 #
fig5, ax5 = plt.subplots(1,1)
ax5.set_xlabel(r"$r \mathrm{ / R_{500}}$", fontsize=16)
ax5.set_ylabel(r"$\mathrm{ \frac{N_r}{A_r/A_R}}$", fontsize=16)

ax5.plot(distance, ncount/Anorm, "ko-", lw=3)
fig5.savefig(savedir + "submm_sources.png", dpi=1200, bbox_inches="tight")


# Figure 6 #
rows, cols = 6, 9
fig6, ax6 = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(12,20))
ax6 = ax6.flatten()
fig6.text(0.5, -0.01, r"$r \mathrm{ / R_{500}}$", ha="center", fontsize=16)
fig6.text(-0.02, 0.5, r"$\mathrm{F_{\lambda} \/\/ \left[ mJy/px \right]}$",
							ha="center", fontsize=16, rotation="vertical")

colors = ["salmon", "orange", "darkturquoise"]
lines = [[]]*len(wavl)  # dummy variable storing Line2D objects
for j in range(rows*cols):
	for i in range(len(wavl)):
		if j < len(fluxind[0]):
			ax6[j].plot(distance, fluxind[i,j], c=colors[i], lw=1)

			# Shows axes
			if ax6[j].is_first_col:
				ax6[j].tick_params(axis="y", labelsize=7)
			if ax6[j].is_last_row():
				ax6[j].tick_params(axis="x", labelsize=7)

		else:  # turns off unused axes
			wav = "$\\mathrm{ %s \/ \\mu m}$" % wavl[i]
			lines[i] = ax6[j].plot(distance, fluxind[i,0], c=colors[i], lw=4, label=wav)
			ax6[j].legend(loc="center", frameon=False, labelspacing=0, fontsize=12)

			ax6[j].tick_params(axis="x", labelsize=7)  # resizes xaxis
			ax6[j].set_frame_on(False)  # hides frame
			ax6[j].yaxis.set_ticks_position("none")  # hides yticks

			# Line of xaxis
			xmin, xmax = ax6[j].get_xaxis().get_view_interval()
			ymin, ymax = ax6[j].get_yaxis().get_view_interval()
			ax6[j].add_artist(Line2D((xmin, xmax), (ymin, ymin), c="k", lw=2))

[ln[0].set_visible(False) for ln in lines]  # hides line plots
fig6.tight_layout(w_pad=0, h_pad=0)
fig6.savefig(savedir + "fluxind.png", dpi=1200, bbox_inches="tight")


# Figure 7 #
fig7, ax71 = plt.subplots(1,1)
ax72 = ax71.twinx()

lbl1 = r"$r \mathrm{ / R_{500}}$"
lbl2 = r"$\mathrm{\langle z \rangle / z_{*}}$"
lbl3 = r"$\mathrm{ N_{z} / N_{tot} }$"

ax71.set_xlabel(lbl1, fontsize=16)
ax71.set_ylabel(lbl2, fontsize=16)
ax72.set_ylabel(lbl3, fontsize=16)

ax71.errorbar(distance, zr_mean, yerr=zr_std, fmt="ko-", elinewidth=1,
										capsize=2, lw=3, label=lbl2)
ax72.plot(distance, IRratio, "k--", lw=3, label=lbl3)
ax71.legend(loc="upper left", fontsize=16, fancybox=True)
ax72.legend(loc="lower right", fontsize=16, fancybox=True)
fig7.savefig(savedir + "z_submm.png", dpi=1200, bbox_inches="tight")

plt.close("all")
"""
#"""
