#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modifies the b-factor column of a PDB file with accuracy scores.
Usage:
    python color_pdb_by_accuracy.py <pdb file> <score_results.txt>
Example:
    python color_pdb_by_accuracy.py our_6xw6.pdb score_results.txt
"""

import os
import sys
import numpy as np


def check_input(args):
    """
    Validate the given arguments for the program and produce actual data from it
    :param args: the given argument
    :return: the file in the first argument and a list of data from the second argument
    """
    if len(args) != 2:  # validate the number of argument given
        sys.stderr.write('Usage: incorrect number of parameters\n')
        sys.stderr.write(__doc__)
        sys.exit(1)

    else:  # validate the the two given argument are existing and readable files
        if not os.path.isfile(args[0]):
            emsg = 'ERROR: File not found or not readable: \'{}\'\n'
            sys.stderr.write(emsg.format(args[0]))
            sys.stderr.write(__doc__)
            sys.exit(1)

        if not os.path.isfile(args[1]):
            emsg = 'ERROR: File not found or not readable: \'{}\'\n'
            sys.stderr.write(emsg.format(args[1]))
            sys.stderr.write(__doc__)
            sys.exit(1)

    fh = open(args[0], 'r')  # open the PDB file
    data = []  # turn the scores file into a list
    with open(args[1]) as file:
        for line in file:
            line = line.strip()
            data.append(float(line))
    return fh, data


def pad_line(line):
    """
    Pad (or truncate) line to 80 characters in case it is shorter, to match the PDB conventions
    :param line: the line to pad
    :return: the padded line
    """
    size_of_line = len(line)
    if size_of_line < 80:
        padding = 80 - size_of_line + 1
        line = line.strip('\n') + ' ' * padding + '\n'
    return line[:81]  # 80 + newline character


def main():
    pdbfh, data = check_input(sys.argv[1:])
    newfh = open(f'accuracy_{sys.argv[1]}', 'w')  # open the output file
    chain = 'H'  # for the nanobodies, the only chain is the heavy chain

    try:
        to_write = []
        for line in pdbfh:
            # the AAs we actually interested in
            if line.startswith('ATOM') and line[21] == chain:
                line = pad_line(line)
                # find score by index (in line[23:26])
                line = line[:60] + "{0:>6.2f}".format(data[int(line[23:26])]) + line[66:]
            # if there are uninvolved chains or atoms
            elif (line.startswith('ATOM') and line[21] != chain) or line.startswith('HETATM'):
                line = pad_line(line)
                line = line[:60] + "100.00" + line[66:]
            to_write.append(line)
        newfh.write(''.join(to_write))
    except IOError:
        sys.stderr.write('IOError, please try to run again\n')
        sys.stderr.write(__doc__)
        sys.exit(1)

    # close all files and exit
    newfh.close()
    pdbfh.close()
    sys.exit(0)


if __name__ == '__main__':
    main()
