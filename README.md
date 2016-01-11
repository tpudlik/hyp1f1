# hyp1f1

Reference values for the confluent hypergeometric function.


## Usage ##

To compute the reference values, run

    python compute_reference.py

To change the locations at which reference values will be computed, edit
`config.py`. To create plots of function accuracy, run

    python make_accuracy_plots.py hyp1f1

or,

    python make_accuracy_plots.py new_hyp1f1


## Interpreting the plots ##

The plots show relative error:

    E = (estimate - true)/|true|

Positive errors are red, negative errors are blue, and locations where the
estimate was NaN of Inf (but the true value was not an Inf with the same sign)
are marked in black.
