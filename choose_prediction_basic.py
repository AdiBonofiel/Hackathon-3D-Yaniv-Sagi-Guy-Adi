# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
import sys
import pandas as pd
import numpy as np

#input: 10 matrices of 140*15
#output:
#    -0-1-2-3-4-5-6-7-8-9-10
# 0
# 1
# 2
# 3
# 4
#.

def calculate_score(rmsds_of_chosen_pred):
    return sum(rmsds_of_chosen_pred) * -1


def get_minimal_std_prediction(rmsds_matrix):
    """

    :param rmsds_matrix: 10 on 10
    :return: the index of the chosen prediction
    """
    stds = rmsds_matrix.std(0)
    min_index = np.argmin(stds)
    # print(stds)
    # print(min_index)
    return min_index

def generate_predictions_matrices(preds):
    """
    :param preds: array of 10 matrices of 140*15
    :return: 10*10 matrix with RMSD between each pair of predictions
    """
    rmsds_matrix = np.zeros((len(preds), len(preds)))
    row = 0
    col = 0
    for p_r in preds:
        for p_c in preds:
            rmsds_matrix[row][col] = np.sqrt(np.mean((p_r - p_c) ** 2))
            col += 1
        col = 0
        row += 1
    # #np.sqrt(np.mean((predictions - targets) ** 2))

    return rmsds_matrix



def read_predictions(preds_paths):
    """
    :param preds_paths: array of 10 paths to 10 140*15 matrices
    :return: array of pandas data frames of those matrices
    """
    preds_arr = []
    for pred in preds_paths:
        cur_pred = pd.read_csv(pred, sep='/t')
        preds_arr.append(cur_pred)
    return preds_arr

if __name__ == '__main__':
    args = sys.argv[1:]
    preds_arr = read_predictions(args)

    ### an example predictions array for testing
    # preds_arr = np.array([[[2, 21, 9], [0, 8, 2]], [[3, 888, 1], [1, 4, 2]], [[0, 0, 1], [2, 6, 3]]])

    x = generate_predictions_matrices(preds_arr)
    chosen_pred_index = get_minimal_std_prediction(x)
    acc = calculate_score(x[chosen_pred_index])
    print("chosen structure prediction:")
    print(preds_arr[chosen_pred_index])
    print("accuracy:", acc)



