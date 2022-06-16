import sys
import pandas as pd
import numpy as np
from Bio.PDB import *
import utils


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



def get_seq_aa(pdb_file, chain_id):
    """
    returns the sequence (String) and a list of all the aa residue objects of the given protein chain.
    :param pdb_file: path to a pdb file
    :param chain_id: chain letter (char)
    :return: sequence, [aa objects]
    """
    # load model
    chain = PDBParser(QUIET=True).get_structure(pdb_file, pdb_file)[0][chain_id]

    aa_residues = []
    seq = ""

    for residue in chain.get_residues():
        aa = residue.get_resname()
        if not is_aa(aa) or not residue.has_id('CA'):  # Not amino acid
            continue
        elif aa == "UNK":  # unknown amino acid
            seq += "X"
        else:
            seq += Polypeptide.three_to_one(residue.get_resname())
        aa_residues.append(residue)

    return seq, aa_residues


def write_rmsd_vec_to_file(rmsd_vec):
    f = open('rmsd_vector_basic.txt', 'w')
    for r in rmsd_vec:
        f.write(str(r))
        f.write('\n')
    f.close()

# def provide_predictions(path_to_PDB):
#
#     NUM_OF_MODELS = 10
#     PATH_TO_MODELS = './models'
#     predictions = []
#     protein_to_predict = utils.generate_input(path_to_PDB)
#     for i in range(NUM_OF_MODELS):
#       model = tf.keras.models.load_model("%s/model%d.tf" %(PATH_TO_MODELS, i))
#       prediction = model.predict(protein_to_predict[np.newaxis,:,:])
#       predictions.append(prediction)
#     return predictions

if __name__ == '__main__':
    """
    argv[1] -> protein pdb file path 
    argv[2:] -> the paths to the prediction (npy files)
    """
    # args = sys.argv[2:]
    # pdb_path = sys.argv[1]
    #args = provide_predictions('./Data/6xw6/6xw6.pdb')
    preds_arr = read_predictions('./Data/predictions.npy')
    seq, _ = get_seq_aa('./Data/6xw6/6xw6.pdb', 'H')
    ### an example predictions array for testing
    # preds_arr = np.array([[[1,2,3,4,5,6,7,8,7,6,5,4,3,2,1],[0,0,9,8,7,6,5,4,3,2,3,4,5,6,7]], [[1,2,3,4,0,6,7,8,7,6,5,8,3,2,1],[4,0,9,8,7,2,5,4,3,1,3,4,5,6,7]]])
    # seq = "AA"
    x = generate_predictions_matrices(preds_arr)
    chosen_pred_index = get_reference(x)
    rmsd_vec = generate_rmsd_per_position_vec(preds_arr[chosen_pred_index], preds_arr, chosen_pred_index, 140)
    write_rmsd_vec_to_file(rmsd_vec)
    write_reference_to_pdb(seq, preds_arr[chosen_pred_index][0], "basic_chosen_reference")








### steps:
## input 10 tables  - 140*15

## RMSD all vd. all results in 10*10 matrix

## LOWEST RMSD is the reference

## each position rmsd vs. the reference and average

## test file with 140 rows with the avg rmsds .

