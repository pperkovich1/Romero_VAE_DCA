#!/bin/bash

# Shell script to extract a poperty value from a config file 
# This script is NOT USED anywhere yet but helpful on chtc

# This script is supposed to be run from the run directory on CHTC 
# > ./get_config_chtc.sh ../config.yaml --working_dir

# In both cases, the CHTC server and the group server, this script should be
# able to find the source directory and the read_config.py script 

python3 "../source/read_config.py" "$@"
