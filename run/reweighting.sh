#!/bin/bash

# Shell script to launch reweighting script

# This script is supposed to be run from the main directory on the group server
# > run/reweighting.sh config.yaml

# The CHTC submit script will drop it in the main directory and it will be run
# via the CHTC execute script
# > ./reweighting.sh config.yaml

# In both cases, the CHTC server and the group server, this script should be
# able to find the source directory and the reweighting.py script and put the
# output in the same working directory


cd source || { echo "Error: Cannot change to source directory" ; exit 1; }
python reweighting.py "$@"
cd ..
