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
tar -zxf VAEs.tar.gz
# create the output directory where we can store stuff to return
mkdir output

# move dataset into sequence_sets folder
# TODO: file name should be an input argument
cp ./processed_cmx_uniref100_90_80_10_100.fasta VAEs/sequence_sets

cd VAEs/run

# just in case our sh files are not executable
chmod +x *.sh

# runmodel.sh needs to be run from home directory
cd ..

# pass all arguments (except the first one) to the python program
./run/runmodel.sh "${@:2}"

# tar up the output directory
cd ..
tar -zcf training_output_"$1".tar.gz ./output 
tar -zcf VAEs_file_tree.tar.gz ./VAEs

# clean up all subdirectories
if [ -f "$TOPDIR_FILE" ]; then
    rm -rf */ # safely delete sub directories
else
    echo "$TOPDIR_FILE does not exist. "
    echo "Error: Not cleaning up sub directories"
fi

# reweighting_output_<msa_filename>.tar.gz should be returned automatically
