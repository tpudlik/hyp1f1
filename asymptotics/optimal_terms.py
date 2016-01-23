"""Plot the optimal number of terms in the asymptotic expansion to keep,
as a function of `a` and `b` at given `z`.

"""

import os

import numpy as np
from numpy import pi
from scipy.special import gamma, rgamma
import matplotlib.pyplot as plt

MAXTERMS = 100


def asymptotic_series(a, b, z, maxterms=MAXTERMS):
    """Compute hyp1f1 using an asymptotic series. This uses DLMF 13.7.2
    and DLMF 13.2.4. Note that the series is divergent (as one would
    expect); this can be seen by the ratio test.

    A modified version of the function from hypergeometric, with different 
    output signature.

    Returns
    -------
    S : (maxterms x 1) ndarray
        An array of asymptotic approximations using 1, 2, ..., maxterms of the
        series.
    A : (maxterms x 1) ndarray
        The successive terms of the asymptotic approximations.

    """
    # S1 is the first sum; the ith term is
    # (1 - a)_i * (b - a)_i * z^(-s) / i!
    # S2 is the second sum; the ith term is
    # (a)_i * (a - b + 1)_i * (-z)^(-s) / i!
    A1 = 1
    S1 = A1
    A2 = 1
    S2 = A2
    A1n = np.arange(maxterms, dtype=np.float64)
    A2n = np.arange(maxterms, dtype=np.float64)
    S1n = np.arange(maxterms, dtype=np.float64)
    S2n = np.arange(maxterms, dtype=np.float64)
    # Is 8 terms optimal? Not sure.
    for i in xrange(1, maxterms + 1):
        A1 = A1*(i - a)*(b - a + i - 1) / (z*i)
        S1 += A1
        A2 = -A2*(a + i - 1)*(a - b + i) / (z*i)
        S2 += A2
        A1n[i - 1] = A1
        A2n[i - 1] = A2
        S1n[i - 1] = S1
        S2n[i - 1] = S2
    
    phi = np.angle(z)
    if np.imag(z) == 0:
        expfac = np.cos(pi*a)
    elif phi > -0.5*pi and phi < 1.5*pi:
        expfac = np.exp(1J*pi*a)
    elif phi > -1.5*pi and phi <= -0.5*pi:
        expfac = np.exp(-1J*pi*a)
    else:
        raise Exception("Shouldn't be able to get here!")

    c1 = np.exp(z)*z**(a - b)*rgamma(a)
    c2 = expfac*z**(-a)*rgamma(b - a)
    return (gamma(b)*(c1*S1n + c2*S2n), gamma(b)*(c1*A1n + c2*A2n))

def optimal_number_of_terms(a, b, z, ref):
    bc = np.broadcast(a, b, z, ref)
    a = np.ones(shape=bc.shape)*a
    b = np.ones(shape=bc.shape)*b
    z = np.ones(shape=bc.shape)*z
    out = np.empty(shape=bc.shape)
    for idx in np.ndindex(bc.shape):
        estimate = asymptotic_series(a[idx], b[idx], z[idx])[0]
        error = estimate - ref[idx]
        try:
            out[idx] = np.nanargmin(np.abs(error))
            print out[idx]
        except ValueError:
            # All-NaN slice
            out[idx] = np.nan
    return out

def smallest_term(a, b, z):
    bc = np.broadcast(a, b, z)
    a = np.ones(shape=bc.shape)*a
    b = np.ones(shape=bc.shape)*b
    z = np.ones(shape=bc.shape)*z
    out = np.empty(shape=bc.shape)
    for idx in np.ndindex(bc.shape):
        estimate = asymptotic_series(a[idx], b[idx], z[idx])[1]
        try:
            out[idx] = np.nanargmin(np.abs(estimate))
        except ValueError:
            out[idx] = np.nan
    return out

def last_decreasing_term(a, b, z):
    bc = np.broadcast(a, b, z)
    a = np.ones(shape=bc.shape)*a
    b = np.ones(shape=bc.shape)*b
    z = np.ones(shape=bc.shape)*z
    out = np.empty(shape=bc.shape)

    for idx in np.ndindex(bc.shape):
        estimate = asymptotic_series(a[idx], b[idx], z[idx])[1]
        smallest_so_far = np.abs(estimate[0])
        smallest_index = 0
        for index, e in np.ndenumerate(estimate):
            if np.abs(e) <= smallest_so_far:
                smallest_so_far = np.abs(e)
                smallest_index = index[0]
            else:
                break
        try:
            out[idx] = smallest_index
        except:
            print smallest_index
            raise
    return out

def first_irrelevant_term(a, b, z):
    bc = np.broadcast(a, b, z)
    a = np.ones(shape=bc.shape)*a
    b = np.ones(shape=bc.shape)*b
    z = np.ones(shape=bc.shape)*z
    out = np.empty(shape=bc.shape)

    for idx in np.ndindex(bc.shape):
        sums, terms = asymptotic_series(a[idx], b[idx], z[idx])
        for index, s in np.ndenumerate(sums):
            if s + terms[index] == s:
                break
        out[idx] = index[0]

    return out

def get_reference_data(idx):
    maindir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fname = os.path.join(maindir, "reference", "data",
                         "reference_data_{:02}.npz".format(int(idx)))
    with open(fname, "rb") as f:
        data = np.load(f)
        return (data['a'], data['b'], data['z'], data['ref'])

def colormap(a, b, z, imdata):
    cmap = plt.cm.Greens
    cmap.set_bad('r', 1)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(a, b, imdata, cmap=cmap, vmin=1, vmax=MAXTERMS)
    plt.colorbar(im)

    ax.set_xlim((np.min(a), np.max(a)))
    ax.set_ylim((np.min(b), np.max(b)))
    ax.set_xlabel("a")
    ax.set_ylabel("b")

    ax.set_title("Term number for z = {:.2e}".format(np.float64(z)))

    return fig

def make_plot(kind, idx, fig_fname):
    a, b, z, ref = get_reference_data(idx)
    if kind == "optimal":
        imdata = optimal_number_of_terms(a, b, z, ref)
    elif kind == "smallest":
        imdata = smallest_term(a, b, z)
    elif kind == "last_decreasing":
        imdata = last_decreasing_term(a, b, z)
    elif kind == "first_irrelevant":
        imdata = first_irrelevant_term(a, b, z)
    else:
        raise ValueError("Unexpected kind argument, {}".format(kind))

    imdata = np.ma.masked_invalid(imdata)
    fig = colormap(a, b, z, imdata)
    plt.savefig(fig_fname)
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("idx", choices=map(str, xrange(31)),
                        help="Index of reference data set to use.")
    parser.add_argument("kind", choices=["smallest", "optimal",
                                         "last_decreasing",
                                         "first_irrelevant"],
                        help="Type of plot to make.")
    args = parser.parse_args()

    make_plot(args.kind, args.idx,
              "{}_terms_{:02}.png".format(args.kind, int(args.idx)))
