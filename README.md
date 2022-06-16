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

## Modified Network - deep learning approach for accuracy predicting

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
