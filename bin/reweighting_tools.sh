#!/bin/bash
# Run the reweighting_tools.py python program as a shell script


# Activate Conda environment
conda_source="/home/romeroroot/miniconda3/bin/activate"
source ${conda_source} pytorch-docker


# Find out where this script is located
# It should be in the bin directory of a VAE installation
SCRIPT=`realpath $0` 
SCRIPTPATH=`dirname $SCRIPT`

# Find out the python source directory for the corresponding
# VAE directory
SOURCEDIR="$SCRIPTPATH/../source" 

# ADD Source to PYTHONPATH
PYTHONPATH="${SOURCEDIR}:${PYTHONPATH}" python "${SOURCEDIR}"/reweighting_tools.py "$@"
