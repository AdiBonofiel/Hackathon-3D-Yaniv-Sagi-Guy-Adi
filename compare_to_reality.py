#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

TEST_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/NbTestSet"
PRED_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/yaniv_s_model_output"
BACKBONE_ATOMS = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}


def main():
    mean_abs_errors = []
    real_positions = [[] for _ in range(116)]
    pred_positions = [[] for _ in range(116)]
    for filename in os.listdir(TEST_DIR):
        ref_fh = open(os.path.join(TEST_DIR, filename), 'r')
        pred_fh = open(os.path.join(PRED_DIR, filename), 'r')
        res = open(os.path.join(PRED_DIR, f"{filename[:-4]}.txt"), 'r')
        ref_lines = ref_fh.readlines()
        ref_lines = [line for line in ref_lines if line[13:15].strip() in BACKBONE_ATOMS]

        try:
            real_res = []
            one_AA = []
            index = -1
            tag = " "
            for ref_line, pred_line in zip(ref_lines, pred_fh):
                if ref_line.startswith('ATOM'):
                    if index == -1:
                        index = int(ref_line[23:26])
                    if int(ref_line[23:26]) != index or tag != ref_line[26]:
                        real_res.append((max(3 - math.sqrt(np.mean(one_AA)), 0) / 3) * 100)
                        one_AA = []
                        index = int(ref_line[23:26])
                        tag = ref_line[26]
                    if ref_line[13:15] != pred_line[13:15]:
                        sys.stderr.write('ERROR: Mismatch between the lines')
                        sys.exit(1)
                    x = (float(ref_line[31:38]) - float(pred_line[31:38])) ** 2
                    y = (float(ref_line[39:46]) - float(pred_line[39:46])) ** 2
                    z = (float(ref_line[47:54]) - float(pred_line[47:54])) ** 2
                    one_AA.append(x + y + z)
            real_res.append((max(3 - math.sqrt(np.mean(one_AA)), 0) / 3) * 100)
            res_lines = res.readlines()
            res_lines = [(max(3 - float(line), 0) / 3) * 100 for line in res_lines]
            mean_error = np.mean(np.abs(np.array(res_lines[:len(real_res)]) - np.array(real_res)))
            mean_abs_errors.append(mean_error)
            for i in range(116):
                real_positions[i].append(real_res[i])
                pred_positions[i].append(res_lines[i])
        except IOError:
            sys.stderr.write('IOError, please try to run again\n')
            sys.stderr.write(__doc__)
            sys.exit(1)

        # close all files and exit
        ref_fh.close()
        pred_fh.close()
        res.close()

    print(np.mean(mean_abs_errors))
    real_stds = []
    real_means = []
    pred_stds = []
    pred_means = []
    for i in range(116):
        real_stds.append(np.std(real_positions[i]))
        real_means.append(np.mean(real_positions[i]))
        pred_stds.append(np.std(pred_positions[i]))
        pred_means.append(np.mean(pred_positions[i]))

    x = np.array([i for i in range(116)])
    plt.rcParams["figure.figsize"] = (12, 5.5)
    pp = []
    p = plt.errorbar(x, np.array(real_means), np.array(real_stds), linestyle='None', marker='.')
    pp.append(p)
    p = plt.errorbar(x, np.array(pred_means), np.array(pred_stds), linestyle='None', marker='.')
    pp.append(p)
    plt.legend(pp, ['Real accuracy', 'Predicted accuracy'], numpoints=1, loc='lower left')
    plt.title("Comparison between predicted and real accuracy for modified network")
    plt.xlabel("Position")
    plt.ylabel("Accuracy score")
    plt.show()


if __name__ == '__main__':
    main()
