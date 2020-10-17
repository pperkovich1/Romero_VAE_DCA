#!/bin/bash

# Shell script to extract a poperty value from a config file 
# This script is NOT USED anywhere yet but helpful on chtc

# This script is supposed to be run from the run directory on CHTC 
# > ./get_config_chtc.sh ../config.yaml --working_dir

# To see the help run the command with argument -h
# > ./get_config_chtc.sh -h

python3 "../source/read_config.py" "$@"
