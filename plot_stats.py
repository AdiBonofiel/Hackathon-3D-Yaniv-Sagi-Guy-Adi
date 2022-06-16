#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot some statistic about the given PDB with accuracy
Usage:
    python plot_stats.py <pdb file>
Example:
    python plot_stats.py accuracy_our_6xw6.pdb
"""
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

CDR1_START = 50
CDR1_END = 57
CDR2_START = 25
CDR2_END = 33
CDR3_START = 98
CDR3_END = 120


def check_input(args):
    """
    Validate the given arguments for the program and produce actual data from it
    :param args: the given argument
    :return: the file in the first argument and a list of data from the second argument
    """
    if len(args) != 1:  # validate the number of argument given
        sys.stderr.write('Usage: incorrect number of parameters\n')
        sys.stderr.write(__doc__)
        sys.exit(1)

    else:  # validate the the two given argument are existing and readable files
        if not os.path.isfile(args[0]):
            emsg = 'ERROR: File not found or not readable: \'{}\'\n'
            sys.stderr.write(emsg.format(args[0]))
            sys.stderr.write(__doc__)
            sys.exit(1)

    return open(args[0], 'r')  # open the PDB file


def main():
    pdbfh = check_input(sys.argv[1:])

    try:
        variable_regions = []
        const_regions = []
        index = -1
        for line in pdbfh:
            if line.startswith('ATOM'):
                if index == int(line[23:26]):
                    continue
                index = int(line[23:26])
                if CDR1_START <= index <= CDR1_END or CDR2_START <= index <= CDR2_END or CDR3_START <= index <= CDR3_END:
                    variable_regions.append(float(line[61:66]))
                else:
                    const_regions.append(float(line[61:66]))
    except IOError:
        sys.stderr.write('IOError, please try to run again\n')
        sys.stderr.write(__doc__)
        sys.exit(1)

    bins = np.arange(min(min(variable_regions), min(const_regions)), 100, 5)  # fixed bin size

    plt.xlim([max(min(min(variable_regions), min(const_regions)) - 5, 0), 105])
    plt.hist(const_regions, bins=bins, alpha=0.5, label="Constant regions")
    plt.hist(variable_regions, bins=bins, alpha=0.5, label="Variable regions")

    plt.title('Constant vs. variable regions accuracy')
    plt.xlabel('The accuracy (bin size = 5)')
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
