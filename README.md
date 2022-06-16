# Hackathon-3D-Yaniv-Sagi-Guy-Adi
Hackathon and fun for 3-D folding course 

# Getting started:

1. After clonning the repository, install requierments:

`pip install -r requirements.txt`

2. run perdiction by:

`python run_prediction.py <path_to_pdb> <method>`

When method is either `baseline` or `advanced`.
 
A pdb file contain the predicted structure and a txt file with confidence level of each residu prediction, will be saved in your home directory.

3. For visualization run : 🖌️

`python color_pdb_by_accuracy.py <pdb_file> <score_results.txt>`
 
where pdb_file is a .PDB file you want to color and score_results.txt is a .txt file with the data for coloring (in our case, the accuracy).
The program outputs a new .PDB file name accuracy_<pdb_file>, which can be colored by accuracy using bfactor (in ChimeraX, write the command "color bfactor"). The output should looks like this: <p align="center"> 

https://user-images.githubusercontent.com/96491832/173994370-ff738297-785e-467e-919d-10ba0cd907b6.mp4

 </p>


# Implementaion details:

## Baseline model - comparing 10 nets approach for accuracy prediction
First, train 10 networks (architecture from ex4) with different parameters  - 10net.ipynb file.
```python
def train_models(num_models_to_train, meta_parametes):
    '''
    meta params: 0.res1_blocks, 1.res1_ker_size, 2.res1_ker_num, 3.activation_res1,
    4.res2_blocks, 5.res2_ker_size, 6.res2_ker_num, 7.activation_res2,
    8.dilaitons, 9.dropout, 10.epoch, 11.batch, 12.activation_dropout 13.LR
    '''
    path_for_models =  "/content/drive/MyDrive/hackathonPrivate/models"
    X = np.load('/content/drive/MyDrive/hackathonPrivate/files/train_input.npy')
    Y = np.load('/content/drive/MyDrive/hackathonPrivate/files/train_labels.npy')

    X_train, X_val, y_train, y_val  = train_test_split(X, Y, test_size=0.2, random_state=1)
    for i in range(num_models_to_train):
        print("model %d" %i)
        if (meta_parametes[i] == []):
          model = build_network()
          my_optimizer= tf.keras.optimizers.Adam(learning_rate=LR)

        else:
          model = build_network(meta_parametes[i][0], meta_parametes[i][1], meta_parametes[i][2], meta_parametes[i][3], meta_parametes[i][4], 
                                meta_parametes[i][5], meta_parametes[i][6],meta_parametes[i][7], meta_parametes[i][8], meta_parametes[i][9], 
                                meta_parametes[i][12])
          my_optimizer= tf.keras.optimizers.Adam(learning_rate=meta_parametes[i][13])
        model.compile(optimizer=my_optimizer, loss="mse")
        batch = BATCH if meta_parametes[i] == [] else meta_parametes[i][11]
        epoch = EPOCHS if meta_parametes[i] == [] else meta_parametes[i][10]
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, validation_data=(X_val, y_val))
        model.save("%s/model%d.tf" %(path_for_models, i))
if __name__ == '__main__':
    #
    #   Activate the train models only if you wish to edit/change/add the models
    #   Edit the meta_parameters to be as you wish in the models  
    #

    meta_parameters = [[],[3, 15, 64, 'relu', 5, 5, 32, 'relu', [1,2,4], 0.2, 50, 32, 'elu', 0.01],
                       [3, 15, 64, 'relu', 3, 5, 24, 'relu', [2,4,8,16], 0.25, 60, 64, 'elu', 0.01],
                       [3, 25, 48, 'relu', 2, 5, 32, 'relu', [1,2,4,8,16], 0.15, 60, 32, 'elu', 0.01],
                       [3, 15, 64, 'relu', 5, 5, 32, 'gelu', [1,2,4,8], 0.2, 60, 32, 'elu', 0.01],
                       [3, 15, 64, 'softplus', 3, 3, 32, 'softplus', [1,2,4,8], 0.15, 60, 32, 'elu', 0.01],
                       [3, 15, 64, 'elu', 5, 3, 64, 'elu', [1,2,4,16], 0.2, 60, 32, 'relu', 0.01],
                       [3, 15, 64, 'relu', 3, 5, 32, 'LeakyReLU', [1,2,4,8], 0.25, 60, 32, 'elu', 0.01], #<----
                       [4, 15, 64, 'silu', 5, 3, 32, 'silu', [1,2,4], 0.2, 50, 64, 'elu', 0.01],
                       [3, 15, 64, 'relu', 2, 5, 64, 'relu', [1,2,4,8,16], 0.25, 120, 32, 'elu', 0.001]]
    train_models(10, meta_parameters)
```
Second, find the chosen prediction of 10 for using as reference - run_prediction.py
```python
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



def get_reference(rmsds_matrix):
    """
    :param rmsds_matrix: 10 on 10
    :return: the index of the chosen prediction
    """
    rmsd_sum = rmsds_matrix.sum(0)
    min_index = np.argmin(rmsd_sum)
    return min_index
    
def write_rmsd_vec_to_file(rmsd_vec, protein_name):
    f = open("baseline_model_" + protein_name + '_acc.txt', 'w')
    for r in rmsd_vec:
        f.write(str(r))
        f.write('\n')
    f.close()
```
Third, generate average RMSD per position and write it to txt file.
```python

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
```

Forth, write chosen reference to pdb file. 
```python
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
```


## Modified Network - deep learning approach for accuracy prediction

First, architecture modification is needed for the network emit two tensors - structure prediction and accuracy prediction:
```python
def build_network():
    """
    builds the neural network architecture as shown in the exercise.
    :return: a Keras Model
    """
    # input, shape (NB_MAX_LENGTH,FEATURE_NUM)
    input_layer = tf.keras.Input(shape=(utils.NB_MAX_LENGTH, utils.FEATURE_NUM))
    # network layers here
    output1 = layers.Dense(utils.OUTPUT_SIZE)(last_layer)
    output2 = layers.Dense(1)(last_layer)
    return tf.keras.Model(input_layer, [output1, output2], name='from_seq_to_struct')

```

Next, we design two losses, one per each output:

```python
def rmsd_loss(y_true, y_pred):
    mse_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    mse = mse_func(y_true, y_pred)
    rmsd = tf.sqrt(((mse * 15) / 5))
    return rmsd 

def loss_acc_fn(y_true, y_pred, coef):
    return coef * rmsd_loss(y_true, y_pred)
```

Now, we can run a training loop. We start to use the second loss after 10 epoches, which is enough time for the network to converge on the structure prediction.

```python
for epoch in range(epochs):
    if epoch >= 10:
        coef = 0.03
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            y_pred, acc_pred = model(x_batch_train, training=True)  
            # Compute the loss value for this minibatch.
            rmsd_loss_value = rmsd_loss(y_batch_train, tf.cast(y_pred, tf.float64))
            acc_loss_value = loss_acc_fn(rmsd_loss_value, tf.cast(acc_pred, tf.float64), coef)
        grads = tape.gradient([rmsd_loss_value, acc_loss_value], model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

