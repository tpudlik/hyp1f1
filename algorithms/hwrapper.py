"""hwrapper: Returns wrappers for the hypergeometric algorithms."""

import logging

import numpy as np
import scipy.special

import hypergeometric

def get_function(name):
    """Return the function to use in computing estimates of hyp1f1.

    Parameters
    ----------
    name : str
        The name of any function in the hypergeometric module, or "hyp1f1" for
        the `scipy.special` function.

    """

    vectorized = ("hyp1f1",)

    try:
        func = getattr(scipy.special, name)
    except:
        func = getattr(hypergeometric, name)

    if name not in vectorized:
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
