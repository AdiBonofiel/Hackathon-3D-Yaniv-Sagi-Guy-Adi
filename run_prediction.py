import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
import import_ipynb
import utils
from Bio.PDB import *
from predictions_provider import provide_predictions


#input: X matrices of 140*15
#generating pdb file of the chosen reference and txt file with the avg RMSD vector:


def generate_rmsd_per_position_vec(reference, others, chosen_pred_index, number_of_positions):
    # print(reference)
    # print(others)
    p = 0
    final_vec = []
    for i in range(number_of_positions):
        cur_pos_vec = []
        for pred in others:
            if p == chosen_pred_index:
                p += 1
                continue
            else:
                cur_pred = pred[0][i]
                cur_ref = reference[0][i]
                cur_pos_vec.append(np.sqrt(np.mean((cur_pred - cur_ref) ** 2)))
                p += 1
        p = 0
        cur_mean = np.mean(cur_pos_vec)
        final_vec.append(cur_mean)

    return final_vec




def get_reference(rmsds_matrix):
    """

    :param rmsds_matrix: 10 on 10
    :return: the index of the chosen prediction
    """
    rmsd_sum = rmsds_matrix.sum(0)
    min_index = np.argmin(rmsd_sum)
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
    # preds_arr = []
    # for pred in preds_paths:
    #     cur_pred = pd.read_csv(pred, sep='/t')
    #     preds_arr.append(cur_pred)
    preds_arr = np.load(preds_paths)
    return preds_arr


def write_reference_to_pdb(seq, coord_matrix, pdb_name):
    """
    Receives a sequence (String) and the output matrix of the neural network (coord_matrix, numpy array)
    and creates from them a PDB file named pdb_name.pdb.
    :param seq: protein sequence (String), with no padding
    :param coord_matrix: output np array of the nanobody neural network, shape = (NB_MAX_LENGTH, OUTPUT_SIZE)
    :param pdb_name: name of the output PDB file (String)
    """
    BACKBONE_ATOMS = ["N", "CA", "C", "O", "CB"]
    ATOM_LINE = "ATOM{}{}  {}{}{} {}{}{}{}{:.3f}{}{:.3f}{}{:.3f}  1.00{}{:.2f}           {}\n"
    END_LINE = "END\n"
    k = 1

    with open(f"{pdb_name}.pdb", "w") as pdb_file:
        for i, aa in enumerate(seq):
            third_space = (4 - len(str(i))) * " "
            for j, atom in enumerate(BACKBONE_ATOMS):
                if not (aa == "G" and atom == "CB"):  # GLY lacks CB atom
                    x, y, z = coord_matrix[i][3 * j], coord_matrix[i][3 * j + 1], coord_matrix[i][3 * j + 2]
                    b_factor = 0.00
                    first_space = (7 - len(str(k))) * " "
                    second_space = (4 - len(atom)) * " "
                    forth_space = (12 - len("{:.3f}".format(x))) * " "
                    fifth_space = (8 - len("{:.3f}".format(y))) * " "
                    sixth_space = (8 - len("{:.3f}".format(z))) * " "
                    seventh_space = (6 - len("{:.2f}".format(b_factor))) * " "

                    pdb_file.write(
                        ATOM_LINE.format(first_space, k, atom, second_space, Polypeptide.one_to_three(aa), "H",
                                         third_space,
                                         i, forth_space, x, fifth_space, y, sixth_space, z, seventh_space,
                                         b_factor, atom[0]))
                    k += 1

        pdb_file.write(END_LINE)


def write_rmsd_vec_to_file(rmsd_vec, protein_name):
    f = open("baseline_model_" + protein_name + '_acc.txt', 'w')
    for r in rmsd_vec:
        f.write(str(r))
        f.write('\n')
    f.close()

def main(args):
  pdb_path = args[1]
  method = args[2]
  run_prediction(pdb_path, method)

def run_prediction(pdb_path, method):
  if method == "baseline":
    preds_arr = provide_predictions(pdb_path)
    protein_name = os.path.basename(os.path.splitext(pdb_path)[0])
    seq, _ = utils.get_seq_aa(pdb_path, 'H')
    x = generate_predictions_matrices(preds_arr)
    chosen_pred_index = get_reference(x)
    rmsd_vec = generate_rmsd_per_position_vec(preds_arr[chosen_pred_index], preds_arr, chosen_pred_index, 140)
    write_rmsd_vec_to_file(rmsd_vec, protein_name)
    write_reference_to_pdb(seq, preds_arr[chosen_pred_index][0], "baseline_model_" + protein_name)
   
  if method == "advanced":
    path_to_advanced_model = "model_hackathon.tf"
    model = tf.keras.models.load_model(path_to_advanced_model)
    protein_name = os.path.basename(os.path.splitext(pdb_path)[0])
    to_predict = utils.generate_input(pdb_path)
    struct, acc = model.predict(to_predict[np.newaxis,:,:])
    utils.matrix_to_pdb(utils.get_seq_aa(pdb_path, "H")[0], struct[0, :, :], "advanced_model_" + protein_name)
    np.savetxt("advanced_model_" + protein_name + '_acc.txt', np.squeeze(acc), fmt='%5f')
  

if __name__ == '__main__':
    main(sys.argv)
