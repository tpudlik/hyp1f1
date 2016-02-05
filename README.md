# hyp1f1

Reference values for the confluent hypergeometric function.


## Usage ##

To compute the reference values, run

    python compute_reference.py

To change the locations at which reference values will be computed, edit
`config.py`. To create plots of function accuracy using the reference
values, run

    python make_accuracy_plots.py hyp1f1

or,

    python make_accuracy_plots.py new_hyp1f1


Diagnostic plots for the asymptotic series can be computed using
`asymptotics/diagnostic_plot.py`.  These plots pertain to a
particular point in parameter space, and so do not require precomputing
the reference values.
