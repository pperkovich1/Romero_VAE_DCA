#!/bin/bash

# Shell script to launch reweighting script
# $1 is the postfix to the output tar.gz file
# $2 onwards are passed onto the python program


# Group server check!!
# if the home directory has a @ in it then we
# might be on the group server so do not execute
# as this script will delete all subdirectories in 
# the parent directory without asking
if [[ $HOME == *"@"* ]]; then 
  echo "This script is only meant to be executed on chtc. Is this the group server?"; 
  exit 1
fi

# set up the staging environment
tar -zxvf staging.tar.gz
# create the output directory where we can store stuff to return
mkdir output

# pass all arguments (except the first one) to the python program
cd source
python reweighting.py "${@:2}"

# tar up the output directory
cd ..
tar -zcvf output_"$1".tar.gz -C output/ .

# clean up all subdirectories
rm -rf */
# output.tar.gz should be returned automatically
