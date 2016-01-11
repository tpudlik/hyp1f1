# Parameters for compute_reference.py

# mpmath maximum precision when computing hypergeometric function values.
MAXPREC = 100000

# Range of the logarithm of a and b.  PTS should be an odd number, since
# a = 0 and b = 0 are included in addition to positive and negative values.
LOWER = -8
UPPER = 2
PTS = 401

# Range of the logarithm of z values.
LOWER_Z = -4
UPPER_Z = 3
PTS_Z = 31
