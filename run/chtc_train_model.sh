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
tar -zxf staging.tar.gz
# create the output directory where we can store stuff to return
mkdir output

# add dataset to config file
printf "\naligned_msa_filename:%s\n" $2 >> config.yaml
 
# move dataset into sequence_sets folder
# TODO: find prettier way to accomplish this
# cp ./processed_cmx_uniref100_90_80_10_100.fasta sequence_sets
cp $2 sequence_sets

# for debuging purposes
ls -aR

# just in case our sh files are not executable
chmod +x *.sh

# pass config file location to the python program
./runmodel.sh $3

# tar up the output directory
tar -zcf training_output_"$1".tar.gz ./output 

# clean up all subdirectories
if [ -f "$TOPDIR_FILE" ]; then
    rm -rf */ # safely delete sub directories
else
    echo "$TOPDIR_FILE does not exist. "
    echo "Error: Not cleaning up sub directories"
fi

# reweighting_output_<msa_filename>.tar.gz should be returned automatically
