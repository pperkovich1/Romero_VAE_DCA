#!/bin/bash

# Shell script to launch reweighting script

# set up the staging environment
tar -zxvf staging.tar.gz
# create the output directory where we can store stuff to return
mkdir output

# pass all arguments to the python program
cd source
python reweighting.py "$@"

# tar up the output directory
cd ..
tar -zcvf output.tar.gz -C output/ .

# clean up all subdirectories
rm -rf */
# output.tar.gz should be returned automatically
