#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

TEST_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/NbTestSet"
PRED_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/preds_10_nets"
PRED2_DIR = "C:/Users/User/Documents/STRUCT_BIO/Hackathon/yaniv_s_model_output"
BACKBONE_ATOMS = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}


def main():
    pred_positions = [[] for _ in range(116)]
    pred_positions2 = [[] for _ in range(116)]
    for filename in os.listdir(TEST_DIR):
        res = open(os.path.join(PRED_DIR, f"{filename[:-4]}.txt"), 'r')
        res2 = open(os.path.join(PRED2_DIR, f"{filename[:-4]}.txt"), 'r')

        try:
            res_lines = res.readlines()
            res_lines = [(max(3 - float(line), 0) / 3) * 100 for line in res_lines]
            res2_lines = res2.readlines()
            res2_lines = [(max(3 - float(line), 0) / 3) * 100 for line in res2_lines]
            for i in range(116):
                pred_positions[i].append(res_lines[i])
                pred_positions2[i].append(res2_lines[i])
        except IOError:
            sys.stderr.write('IOError, please try to run again\n')
            sys.exit(1)

        # close all files and exit
        res.close()
        res2.close()

    pred_stds = []
    pred_means = []
    pred2_stds = []
    pred2_means = []
    for i in range(116):
        pred_stds.append(np.std(pred_positions[i]))
        pred_means.append(np.mean(pred_positions[i]))
        pred2_stds.append(np.std(pred_positions2[i]))
        pred2_means.append(np.mean(pred_positions2[i]))

    x = np.array([i for i in range(116)])
    plt.rcParams["figure.figsize"] = (12, 5.5)
    pp = []
    p = plt.errorbar(x, np.array(pred2_means), np.array(pred2_stds), linestyle='None', marker='.')
    pp.append(p)
    p = plt.errorbar(x, np.array(pred_means), np.array(pred_stds), linestyle='None', marker='.')
    pp.append(p)
    plt.legend(pp, ['Modified network', 'Baseline model'], numpoints=1, loc='lower left')
    plt.title("Comparison between modified network and baseline model accuracy prediction")
    plt.xlabel("Position")
    plt.ylabel("Predicted accuracy score")
    plt.show()


if __name__ == '__main__':
    main()
