"""Nikolay Mayorov's pure-Python arbitrary precision implementation of
hyp1f1.

"""

from decimal import Decimal, getcontext
import numpy as np

try:
    from math import isfinite
except ImportError:
    # isfinite only exists in Python 3; in Python 2 we have to roll our own.
    from math import isnan, isinf

    def isfinite(x):
        return not isnan(x) and not isinf(x)


EPS = np.finfo(float).eps


class ComplexDecimal(object):
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    @classmethod
    def from_complex(cls, x):
        return cls(Decimal(x.real), Decimal(x.imag))

    @classmethod
    def from_double(cls, x):
        return cls(Decimal(x), 0)

    def __mul__(self, other):
        return ComplexDecimal(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real)

    def __imul__(self, other):
        real = self.real
        imag = self.imag
        self.real *= other.real
        self.real -= imag * other.imag
        self.imag *= other.real
        self.imag += real * other.imag
        return self

    def __add__(self, other):
        return ComplexDecimal(self.real + other.real,
                              self.imag + other.imag)

    def __iadd__(self, other):
        self.real += other.real
        self.imag += other.imag
        return self

    def __sub__(self, other):
        return ComplexDecimal(self.real - other.real,
                              self.imag - other.imag)

    def __isub__(self, other):
        self.real -= other.real
        self.imag -= other.imag
        return self

    def __truediv__(self, other):
        real = self.real * other.real + self.imag * other.imag
        imag = self.imag * other.real - self.real * other.imag
        d = other.mag_squared()
        return ComplexDecimal(real / d, imag / d)

    def __itruediv__(self, other):
        self *= other.conj()
        d = other.mag_squared()
        self.real /= d
        self.imag /= d
        return self

    def mag_squared(self):
        return self.real**2 + self.imag**2

    def conj(self):
        return ComplexDecimal(self.real, -self.imag)

    def __complex__(self):
        return complex(self.real, self.imag)


def hyp1f1_taylor_real(a, b, z, tol, maxiter, prec):
    a = Decimal(a)
    b = Decimal(b)
    z = Decimal(z)
    x = Decimal(1)
    s = Decimal(1)

    for i in range(1, maxiter):
        x *= a * z / (i * b)
        s += x

        x_mag = abs(float(x))
        s_mag = abs(float(s))
        if isfinite(x_mag) and isfinite(s_mag) and x_mag < tol * s_mag:
            break

        a += 1
        b += 1

    return float(s)


def hyp1f1_taylor_complex(a, b, z, tol, maxiter, prec):
    a = ComplexDecimal.from_complex(complex(a))
    b = ComplexDecimal.from_complex(complex(b))
    z = ComplexDecimal.from_complex(complex(z))
    one = ComplexDecimal.from_double(1)
    x = ComplexDecimal.from_double(1)
    s = ComplexDecimal.from_double(1)

    for i in range(1, maxiter):
        j = ComplexDecimal.from_double(i)
        x *= a * z / (j * b)
        s += x

        s_mag = float(s.mag_squared())**0.5
        x_mag = float(x.mag_squared())**0.5
        if isfinite(x_mag) and isfinite(s_mag) and x_mag < tol * s_mag:
            break

        a += one
        b += one

    return complex(s)


def hyp1f1(a, b, z, tol=EPS, maxiter=1000, prec=200):
    orig_prec = getcontext().prec
    getcontext().prec = prec

    a = complex(a)
    b = complex(b)
    z = complex(z)

    if b.imag == 0 and b.real <= 0 and b.real == int(b.real):
        return np.nan

    if a.imag == b.imag == z.imag == 0:
        ret = hyp1f1_taylor_real(a.real, b.real, z.real, tol, maxiter, prec)
    else:
        ret = hyp1f1_taylor_complex(a, b, z, tol, maxiter, prec)

    getcontext().prec = orig_prec

    return ret
