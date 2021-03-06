#!/bin/bash

# Shell script to run the VAE model

# This script is supposed to be run from the main directory on the group server
# > run/runmodel.sh config.yaml

# The CHTC submit script will drop it in the main directory and it will be run
# via the CHTC execute script
# > ./runmodel.sh config.yaml

# In both cases, the CHTC server and the group server, this script should be
# able to find the source directory and the reweighting.py script and put the
# output in the same working directory

cd source || { echo "Error: Cannot change to source directory" ; exit 1; }
python vae.py "$@"
python vae_latent_space.py "$@"
cd ..
