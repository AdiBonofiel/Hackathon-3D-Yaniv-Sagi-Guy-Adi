# Hackathon-3D-Yaniv-Sagi-Guy-Adi
Hackathon and fun for 3-D folding course 

# User manual:

### For color_pdb_by_accuracy.py: 🖌️
Run <p align="center"> python color_pdb_by_accuracy.py <pdb_file> <score_results.txt> </p> where pdb_file is a .PDB file you want to color and score_results.txt is a .txt file with the data for coloring (in our case, the accuracy). The program outputs a new .PDB file name accuracy_<pdb_file>, which can be colored by accuracy using bfactor (in ChimeraX, write the command "color bfactor"). The output should looks like this: <p align="center"> <img src ="https://user-images.githubusercontent.com/96491832/173784298-69204751-811c-4089-9fd8-d4113a1fbd3e.png" data-canonical-src="https://user-images.githubusercontent.com/96491832/173784298-69204751-811c-4089-9fd8-d4113a1fbd3e.png" width="300" height="400" /> </p>
