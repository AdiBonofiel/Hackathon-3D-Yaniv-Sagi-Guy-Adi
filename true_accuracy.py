#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import sys
import numpy as np

BACKBONE_ATOMS = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}


def check_input(args):
    """
    Validate the given arguments for the program and produce actual data from it
    :param args: the given argument
    :return: the file in the first argument and a list of data from the second argument
    """
    if len(args) != 2:  # validate the number of argument given
        sys.stderr.write('Usage: incorrect number of parameters\n')
        sys.exit(1)

    else:  # validate the the two given argument are existing and readable files
        if not os.path.isfile(args[0]):
            emsg = 'ERROR: File not found or not readable: \'{}\'\n'
            sys.stderr.write(emsg.format(args[0]))
            sys.exit(1)

        if not os.path.isfile(args[1]):
            emsg = 'ERROR: File not found or not readable: \'{}\'\n'
            sys.stderr.write(emsg.format(args[1]))
            sys.exit(1)

    ref_fh = open(args[0], 'r')
    pred_fh = open(args[1], 'r')
    return ref_fh, pred_fh


def main():
    ref_fh, pred_fh = check_input(sys.argv[1:])
    results = open(f'{ref_fh[:-4]}_results.txt', 'w')  # open the output file
    ref_lines = ref_fh.readlines()
    ref_lines = [line for line in ref_lines if line[13:15].strip() in BACKBONE_ATOMS]

    try:
        to_write = []
        one_AA = []
        index = 1
        for ref_line, pred_line in zip(ref_lines, pred_fh):
            if ref_line.startswith('ATOM'):  # and ref_line[13:15] == 'CA'
                if int(ref_line[23:26]) != index:
                    to_write.append((max(3 - math.sqrt(np.mean(one_AA)), 0) / 3) * 100)
                    one_AA = []
                    index = int(ref_line[23:26])
                if ref_line[13:15] != pred_line[13:15]:
                    sys.stderr.write('ERROR: Mismatch between the lines')
                    sys.exit(1)
                x = (float(ref_line[31:38]) - float(pred_line[31:38])) ** 2
                y = (float(ref_line[39:46]) - float(pred_line[39:46])) ** 2
                z = (float(ref_line[47:54]) - float(pred_line[47:54])) ** 2
                one_AA.append(x + y + z)
        to_write.append((max(3 - math.sqrt(np.mean(one_AA)), 0) / 3) * 100)
        to_write = [str(i) + '\n' for i in to_write]
        results.write(''.join(to_write))
    except IOError:
        sys.stderr.write('IOError, please try to run again\n')
        sys.stderr.write(__doc__)
        sys.exit(1)

    # close all files and exit
    ref_fh.close()
    pred_fh.close()
    results.close()
    sys.exit(0)


if __name__ == '__main__':
    main()
