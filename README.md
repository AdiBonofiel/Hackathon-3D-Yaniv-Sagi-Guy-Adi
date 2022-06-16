# Hackathon-3D-Yaniv-Sagi-Guy-Adi
Hackathon and fun for 3-D folding course 

# Getting started:

1. After clonning the repository, install requierments:

`pip install -r requirments.txt`

2. run perdiction by:

`python run_prediction.py <path_to_pdb> <method>`

 When method is either `baseline` or `advanced`.
 
 A pdb file contain the predicted structure and a txt file with confidence level of each residu prediction, will be saved in your home directory.




### For getting the basic model RMSD vector and chosen structure PDB file:
1. Run choose_prediction_basic.py with argument <path_to_pdb>
2. 2 files will be generated: 
   - basic_chosen_reference.pdb - the pdb of the chosen prediction.
   - rmsd_vector_basic.txt - the mean rmsd of all the predictions with the chosen one per position. 


### For color_pdb_by_accuracy.py: 🖌️
Run <p align="center"> python color_pdb_by_accuracy.py <pdb_file> <score_results.txt> </p> where pdb_file is a .PDB file you want to color and score_results.txt is a .txt file with the data for coloring (in our case, the accuracy). The program outputs a new .PDB file name accuracy_<pdb_file>, which can be colored by accuracy using bfactor (in ChimeraX, write the command "color bfactor"). The output should looks like this: <p align="center"> 

https://user-images.githubusercontent.com/96491832/173994370-ff738297-785e-467e-919d-10ba0cd907b6.mp4

 </p>


