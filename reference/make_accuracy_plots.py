"""Create plots showing regions of low accuracy in evaluating hyp1f1 in
(a, b) space.

"""

import os, sys

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from hyp1f1.algorithms import hypergeometric

# Path hack to allow relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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
    imdata = np.ma.masked_invalid(relative_error(ref, func(a, b, z)))

    cmap = plt.cm.RdYlBu
    cmap.set_bad('k', 1)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(a, b, imdata, cmap=cmap, vmin=-10, vmax=10)
    plt.colorbar(im)

    ax.set_xlim((np.min(a), np.max(a)))
    ax.set_ylim((np.min(b), np.max(b)))

    ax.set_xlabel("a")
    ax.set_ylabel("b")

    ax.set_title("Relative error for z = {:.2e}".format(np.float64(z)))

    return fig

def relative_error(true, estimate):
    out = (estimate - true)/np.abs(true)
    out[true == estimate] = 0  # For np.inf
    return out

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


def get_function(name):
    """Return the function to use in computing estimates of hyp1f1."""

    try:
        func = getattr(scipy.special, args.func)
    except:
        func = getattr(hypergeometric, args.func)

    if args.func == "new_hyp1f1":
        # Eventually new_hyp1f1 will be vectorized, too---but not yet.
        @np.vectorize
        def f(a, b, z):
            try:
                return func(a, b, z)
            except Exception as e:
                msg = "Exception encountered at a = {}, b = {}, z = {}"
                logging.error(msg.format(a, b, z))
                logging.error(e)
                return np.nan

        return f
    else:
        return func


if __name__ == '__main__':
    import argparse, logging, glob

    parser = argparse.ArgumentParser()
    parser.add_argument("func",
                        help="The implementation of hyp1f1 to evaluate",
                        choices=["hyp1f1", "new_hyp1f1"])
    args = parser.parse_args()
    func = get_function(args.func)

    logging.basicConfig(filename='accuracy_plot.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M')
    logging.info("Beginning plot creation...")

    if not os.path.isdir("plots"):
        os.mkdir("plots")
    for data_fname in glob.glob("data/*.npz"):
        fig_fname = (args.func + "_" +
                     os.path.basename(data_fname).split('.')[0] + '.png')
        make_plot(data_fname, func, os.path.join("plots", fig_fname))
        logging.info("Created plot from dataset {}".format(data_fname))

    logging.info("Done!")
