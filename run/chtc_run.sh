#!/bin/bash

# Shell script to launch run script
# $1 is the postfix to the output tar.gz file
# $2 onwards are passed onto the Makefile in the top directory

CHTC_WORK_DIR="chtc_work"

# This is where we will extract all the sequences and source code
# Subdirectories in chtc are not returned so we do not need to delete
# anything in this directory after we are done. 
mkdir -p "${CHTC_WORK_DIR}"

mv staging.tar.gz sequences.tar.gz "${CHTC_WORK_DIR}"
cd "${CHTC_WORK_DIR}"

# set up the staging environment
tar -zxvf staging.tar.gz
tar -zxvf sequences.tar.gz

# by default WORKINGDIR is '../working'
WORKINGDIR=`python source/read_config.py config.yaml --working_dir`
# remove the ../ from the directory name so that WORKINGDIR_NOPARENT is
# 'working'
WORKINGDIR_NOPARENT=${WORKINGDIR#../} 
if [ -z "${WORKINGDIR_NOPARENT}" ];
then
    echo "Working directory not found in config file"
    exit 1
fi

# create the output directory where we can store stuff to return
mkdir -p ${WORKINGDIR_NOPARENT}

# just in case our sh files are not executable
chmod +x run/*.sh

# pass all arguments (except the first one) to the python program
make CONFIG=config.yaml CONFIGDCA=config.yaml CONFIG2d=config.yaml \
        CHECK_CONDA=0 "${@:2}"


MODEL_NAME=`python source/read_config.py config.yaml --model_name`
# tar up the output directory
OUTPUT_TAR_GZ=output_"${MODEL_NAME}".tar.gz
tar -zcvf ../"${OUTPUT_TAR_GZ}"  -C "${WORKINGDIR_NOPARENT}"/ .

# output_<model_name>.tar.gz should be returned automatically
