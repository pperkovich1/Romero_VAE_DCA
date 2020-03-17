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
# create the output directory where we can store stuff to return
mkdir output

# pass all arguments (except the first one) to the python program
./reweighting.sh "${@:2}"

# tar up the output directory
cd ..
tar -zcvf output_"$1".tar.gz -C output/ .

# clean up all subdirectories
if [ -f "$TOPDIR_FILE" ]; then
    rm -rf */ # safely delete sub directories
else
    echo "$TOPDIR_FILE does not exist. "
    echo "Error: Not cleaning up sub directories"
fi

# output.tar.gz should be returned automatically
