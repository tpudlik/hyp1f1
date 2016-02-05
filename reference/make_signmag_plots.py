"""Create plots of the sign and magnitude of hyp1f1 in (a, b) space.

"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def signmag_plot(a, b, z, ref):
    imdata1 = np.sign(ref)
    cmap1 = plt.cm.RdBu
    cmap1.set_bad('k', 1)

    imdata2 = np.log10(np.abs(ref))
    cmap2 = plt.cm.YlOrRd
    cmap2.set_bad('k', 1)

    fig, axarr = plt.subplots(ncols=2, figsize=(12, 6))
    axarr[0].pcolormesh(a, b, imdata1, cmap=cmap1, vmin=-1, vmax=1)
    im = axarr[1].pcolormesh(a, b, imdata2, cmap=cmap2,
                             vmin=np.percentile(imdata2,  5),
                             vmax=np.percentile(imdata2, 95))

    for ax in axarr:
        ax.set_xlim((np.min(a), np.max(a)))
        ax.set_ylim((np.min(b), np.max(b)))
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set(adjustable='box-forced', aspect='equal')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    axarr[0].set_title("Sign of hyp1f1")
    axarr[1].set_title("Magnitude of hyp1f1")
    plt.suptitle("z = {:.2e}".format(np.float64(z)))

    return fig


def make_plot(ref_file, fig_fname):
    with np.load(ref_file) as data:
        a = data['a']
        b = data['b']
        z = data['z']
        ref = data['ref']

    fig = signmag_plot(a, b, z, ref)
    plt.savefig(fig_fname)
    plt.close(fig)

if __name__ == '__main__':
    if not os.path.isdir("signmag_plots"):
        os.mkdir("signmag_plots")

    for data_fname in glob.glob("data/*.npz"):
        fig_fname = ("signmag_" +
                     os.path.basename(data_fname).split('.')[0] + '.png')
        make_plot(data_fname, os.path.join("signmag_plots", fig_fname))
