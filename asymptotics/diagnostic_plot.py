import os
import sys
import argparse

import numpy as np
import mpmath
import matplotlib.pyplot as plt
from matplotlib import colors

import optimal_terms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reference.make_accuracy_plots import correct_digits

RED = (202.0/255, 0, 32.0/255)
GREEN = (0, 109.0/255, 44.0/255)
BLUE = (5.0/255, 113.0/255, 176.0/255)

def make_plot(a, b, z, title, maxterms):
    approx, terms = optimal_terms.asymptotic_series(a, b, z, maxterms)
    ref = np.float64(mpmath.hyp1f1(a, b, z))

    cd = correct_digits(approx, ref)
    termsize = np.abs(terms/terms[0])

    fig, ax1 = plt.subplots()
    ax1.plot(cd, '-', linewidth=2, color=GREEN)
    ax1.set_ylim(0, 17)
    ax1.set_xlabel('term number')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('correct digits', color=GREEN)
    for tl in ax1.get_yticklabels():
        tl.set_color(GREEN)

    ax2 = ax1.twinx()
    cmap, norm = colors.from_levels_and_colors([-np.inf, 0, np.inf],
                                               [BLUE, RED])
    ax2.scatter(np.arange(termsize.shape[0]), termsize,
                c=terms, cmap=cmap, norm=norm, edgecolors='')
    # ax2.semilogy(termsize, 'r--', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_ylabel('relative term size', color=RED)
    # Set the limits, with a little margin.
    ax2.set_xlim(0, termsize.shape[0])
    ax2.set_ylim(np.min(termsize), np.max(termsize))
    for tl in ax2.get_yticklabels():
        tl.set_color(RED)

    ax1.set_title("a = {:.2e}, b = {:.2e}, z = {:.2e}".format(a, b, z))

    plt.savefig("{}.png".format(title))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="The `a` parameter.", type=float)
    parser.add_argument("b", help="The `b` parameter.", type=float)
    parser.add_argument("z", help="The argument.", type=float)
    parser.add_argument("title", help="The title of the plot.")
    parser.add_argument("--maxterms", help="The number of terms to sum.",
                        type=int, default=100)
    args = parser.parse_args()


    make_plot(args.a, args.b, args.z, args.title, args.maxterms)
