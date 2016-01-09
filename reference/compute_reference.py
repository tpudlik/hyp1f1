import logging
import mpmath
import numpy as np

from config import DPS_START, DPS_STEP, DPS_MAX

def reference_array(z, lower=-8, upper=3, pts=30):
    values = pm_logspace(lower, upper, pts)
    a_array, b_array = np.meshgrid(values, values)

    ref = np.empty_like(a_array)

    for idx in np.ndindex(*a_array.shape):
        ref[idx] = reference_value(a_array[idx], b_array[idx], z)

    return (a_array, b_array, ref)

def pm_logspace(start, stop, *args, **kwargs):
    v = 10**np.linspace(start, stop, *args, **kwargs)
    return np.hstack((-v[::-1], 0, v))

def reference_value(a, b, z):
    dps = DPS_START
    mpmath.mp.dps = dps
    try:
        lo = mpmath.hyp1f1(a, b, z)
    except ZeroDivisionError:
        # Pole in hypergeometric series
        return np.inf

    while dps < DPS_MAX:
        dps = dps + DPS_STEP
        mpmath.mp.dps = dps
        hi = mpmath.hyp1f1(a, b, z)

        if np.float64(lo) == np.float(hi):
            return np.float64(hi)
        lo = hi

    logging.warning("No convergence at max precision at ({}, {}, {}).".format(a, b, z))
    return lo


if __name__ == '__main__':
    logging.basicConfig(filename='reference_values.log', level=logging.DEBUG)
    logging.info("Beginning computation of reference values...")

    for idx, z in np.ndenumerate(pm_logspace(-8, 3, 10)):
        a, b, ref = reference_array(z)
        logging.info("Computed values at z = {}".format(z))
        with open("reference_data_{:02}.npz".format(idx[0]), "wb") as f:
            np.savez(f, a=a, b=b, ref=ref, z=z)

    logging.info("Done!")
