#!/bin/bash

# Shell script to launch reweighting script
# $1 is the postfix to the output tar.gz file
# $2 onwards are passed onto the python program

TOPDIR_FILE=chtc_root.txt
if [ -f "$TOPDIR_FILE" ]; then
    echo "$TOPDIR_FILE exist"
else
    echo "$TOPDIR_FILE does not exist"
    echo "This file is needed so that we do not accidentally delete directories "
    exit 1
fi

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
chmod +x *.sh

# pass all arguments (except the first one) to the python program
./run/reweighting.sh "${@:2}"

# tar up the output directory
tar -zcvf reweighting_output_"$1".tar.gz -C ${WORKINGDIR_NOPARENT}/ .

# clean up all subdirectories
if [ -f "$TOPDIR_FILE" ]; then
    rm -rf */ # safely delete sub directories
    rm -f config.yaml 
else
    echo "$TOPDIR_FILE does not exist. "
    echo "Error: Not cleaning up sub directories"
fi

# reweighting_output_<msa_filename>.tar.gz should be returned automatically
