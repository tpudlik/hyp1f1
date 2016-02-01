"""Compute reference values of hyp1f1.

"""

import logging, os

import mpmath
import numpy as np

from config import MAXPREC, UPPER, PTS, LOWER_Z, UPPER_Z, PTS_Z


def reference_array(z, upper=3, pts=201):
    values = np.linspace(-10**upper, 10**upper, pts)
    a_array, b_array = np.meshgrid(values, values)

    ref = np.empty_like(a_array)

    for idx in np.ndindex(*a_array.shape):
        ref[idx] = reference_value(a_array[idx], b_array[idx], z)

    return (a_array, b_array, ref)

def pm_logspace(start, stop, num):
    v = 10**np.linspace(start, stop, (num - 1)/2)
    return np.hstack((-v[::-1], 0, v))

def reference_value(a, b, z):
    mpmath.mp.dps = 20
    try:
        lo = mpmath.hyp1f1(a, b, z, maxprec=MAXPREC)
    except ZeroDivisionError:
        # Pole in hypergeometric series
        return np.inf

    return np.float64(lo)


if __name__ == '__main__':
    logging.basicConfig(filename='reference_values.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M')
    logging.info("Beginning computation of reference values...")

    if not os.path.isdir("data"):
        os.mkdir("data")

    for idx, z in np.ndenumerate(pm_logspace(LOWER_Z, UPPER_Z, PTS_Z)):
        a, b, ref = reference_array(z, UPPER, PTS)
        logging.info("Computed values at z = {}".format(z))
        with open("data/reference_data_{:02}.npz".format(idx[0]), "wb") as f:
            np.savez(f, a=a, b=b, ref=ref, z=z)

    logging.info("Done!")
