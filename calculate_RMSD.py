#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

TEST_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/NbTestSet"
PRED_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/ex4_predictions"
OLD_PRED_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/preds_10_nets"
BACKBONE_ATOMS = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}


def main():
    new_RMSDs = []
    old_RMSDs = []
    for filename in os.listdir(TEST_DIR):
        ref_fh = open(os.path.join(TEST_DIR, filename), 'r')
        pred_fh = open(os.path.join(PRED_DIR, filename), 'r')
        old_pred_fh = open(os.path.join(OLD_PRED_DIR, filename), 'r')
        ref_lines = ref_fh.readlines()
        ref_lines = [line for line in ref_lines if line[13:15].strip() in BACKBONE_ATOMS]

        try:
            real_res = []
            old_res = []
            for ref_line, pred_line, old_line in zip(ref_lines, pred_fh, old_pred_fh):
                if ref_line.startswith('ATOM'):
                    if ref_line[13:15] != pred_line[13:15]:
                        sys.stderr.write('ERROR: Mismatch between the lines')
                        sys.exit(1)
                    x = (float(ref_line[31:38]) - float(pred_line[31:38])) ** 2
                    y = (float(ref_line[39:46]) - float(pred_line[39:46])) ** 2
                    z = (float(ref_line[47:54]) - float(pred_line[47:54])) ** 2
                    real_res.append(x + y + z)
                    x = (float(ref_line[31:38]) - float(old_line[31:38])) ** 2
                    y = (float(ref_line[39:46]) - float(old_line[39:46])) ** 2
                    z = (float(ref_line[47:54]) - float(old_line[47:54])) ** 2
                    old_res.append(x + y + z)
            new_RMSDs.append(math.sqrt(np.mean(real_res)))
            old_RMSDs.append(math.sqrt(np.mean(old_res)))
        except IOError:
            sys.stderr.write('IOError, please try to run again\n')
            sys.stderr.write(__doc__)
            sys.exit(1)

        # close all files and exit
        ref_fh.close()
        pred_fh.close()
        old_pred_fh.close()

    fig, ax = plt.subplots()
    plt.scatter(new_RMSDs, old_RMSDs)
    plt.title("Comparison of RMSD between ex4 network and baseline model")
    plt.xlabel(r"RMSD for ex4 network ($\AA$)")
    plt.ylabel(r"RMSD for baseline model (10 networks) ($\AA$)")
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.show()


if __name__ == '__main__':
    main()
