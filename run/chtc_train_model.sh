#!/bin/bash

# Shell script to launch reweighting script
Cluster=$1
Process=$2
config=$3
# dataset=$3
# hidden=$4
# latent=$5
# config=$6

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
tar -zxf sequences.tar.gz
# create the output directory where we can store stuff to return
mkdir output

# because 'config' is written from perspective in source
cd source
cp "$config" ../output
cd ..

# TODO: make editing config file less janky
# TODO: it'd be nice if I could replace ../config.yaml with $config
# python modify_config.py $(Process) $(config)
# add dataset to config file
# printf "\naligned_msa_filename: %s\n" $dataset >> config.yaml
# printf "\nhidden_layer_size: %s\n" $hidden >> config.yaml
# printf "\nlatent_layer_size: %s\n" $latent >> config.yaml
# cat config.yaml
 
# for debuging purposes
ls -aR

# just in case our sh files are not executable
chmod +x run/*.sh

# pass config file location to the python program
run/runmodel.sh $config

# for debuging
ls -aR
# tar up the output directory
mv ./output ./output_"$Process" #rename output
tar -zcf training_output_"$Process".tar.gz ./output_"$Process"

# clean up all subdirectories
if [ -f "$TOPDIR_FILE" ]; then
    rm -rf */ # safely delete sub directories
else
    echo "$TOPDIR_FILE does not exist. "
    echo "Error: Not cleaning up sub directories"
fi

# reweighting_output_<msa_filename>.tar.gz should be returned automatically
