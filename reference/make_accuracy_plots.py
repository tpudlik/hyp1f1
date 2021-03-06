"""Create plots showing regions of low accuracy in evaluating hyp1f1 in
(a, b) space.

"""

import os, sys

import numpy as np
import matplotlib.pyplot as plt

# Path hack to allow relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms import hwrapper


def accuracy_plot(a, b, z, ref, func):
    """Plot the relative error in using func to approximate hyp1f1.

    Parameters
    ----------
    a, b: array_like
        Meshgrid of a and b values.
    z : float
        Value of the argument.
    ref : array_like
        An array of reference values, of the same shape as a and b.
    func : function
        Function to evaluate.

    Returns
    -------
    Matplotlib Figure object.

    """
    imdata = np.ma.masked_invalid(correct_digits(ref, func(a, b, z)))

    cmap = plt.cm.Greens
    cmap.set_bad('r', 1)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(a, b, imdata, cmap=cmap, vmin=0, vmax=17)
    plt.colorbar(im)

    ax.set_xlim((np.min(a), np.max(a)))
    ax.set_ylim((np.min(b), np.max(b)))
    ax.set_xlabel("a")
    ax.set_ylabel("b")

    ax.set_title("Correct digits for z = {:.2e}".format(np.float64(z)))

    return fig

@np.vectorize
def correct_digits(true, estimate):
    if np.isfinite(true) and not np.isfinite(estimate):
        return 0
    if not np.isfinite(true) and np.isfinite(estimate):
        return 0
    if true == estimate or np.isnan(true) and np.isnan(estimate):
        # both +-inf or nan
        return 17
    if true == 0:
        return 0

    relative_error = np.abs((estimate - true)/true)
    if not np.isfinite(relative_error) or relative_error > 1:
        return 0

    return -np.log10(relative_error)

def make_plot(ref_file, func, fig_fname):
    """Load reference data from ref_file and save an accuracy_plot of func 
    under fig_fname.

    """
    with np.load(ref_file) as data:
        a = data['a']
        b = data['b']
        z = data['z']
        ref = data['ref']

    fig = accuracy_plot(a, b, z, ref, func)
    plt.savefig(fig_fname)
    plt.close(fig)


if __name__ == '__main__':
    import argparse, logging, glob

    parser = argparse.ArgumentParser()
    parser.add_argument("func",
                        help="The implementation of hyp1f1 to evaluate",
                        choices=["hyp1f1", "new_hyp1f1", "taylor_series",
                                 "taylor_series_frac", "single_fraction",
                                 "bessel_series", "asymptotic_series",
                                 "asymptotic_series_muller",
                                 "taylor_series_recur",
                                 "asymptotic_series_full",
                                 "taylor_series_ab_recur",
                                 "rational_approximation",
                                 "old_asymptotic_series",
                                 "hyp1f1_decimal",
                                 "asymptotic_recur"])
    parser.add_argument("--dataset",
                        help="The number of the _single_ dataset to plot",
                        default=False, type=str)
    args = parser.parse_args()
    func = hwrapper.get_function(args.func)

    logging.basicConfig(filename='accuracy_plot.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M')
    logging.info("Beginning plot creation...")

    if not os.path.isdir("plots"):
        os.mkdir("plots")
    

    if args.dataset:
        data_fnames = ["data/reference_data_{}.npz".format(args.dataset)]
    else:
        data_fnames = glob.glob("data/*.npz")

    for data_fname in data_fnames:
        fig_fname = (args.func + "_" +
                     os.path.basename(data_fname).split('.')[0] + '.png')
        make_plot(data_fname, func, os.path.join("plots", fig_fname))
        logging.info("Created plot from dataset {}".format(data_fname))

    logging.info("Done!")
