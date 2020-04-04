#!/bin/bash

# Shell script to plot the latent space of a model with latent dimension 2

# This script is supposed to be run from the main directory on the group server
# > run/plotlatent.sh config2d.yaml 

# The CHTC submit script will drop it in the main directory and it will be run
# via the CHTC execute script
# > ./plotlatent.sh config2d.yaml 

# In both cases, the CHTC server and the group server, this script should be
# able to find the source directory and the reweighting.py script and put the
# output in the same working directory


cd source || { echo "Error: Cannot change to source directory" ; exit 1; }
python reweighting.py "$@"
python vae.py "$@"
python examine_model.py "$@"
python plot_sequences.py "$@"
cd ..
