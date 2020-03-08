#!/bin/bash

# Shell script to launch reweighting script


# Read in the dataset name and threshold
# We pick out the first line that doesn't start with a #
DATASET="dataset.txt"
IFS="," read -r inputfasta threshold \
        < <( cat "${DATASET}" | grep -v "^#" | head -1)

echo "Input Fasta =" $inputfasta
echo "Threshold =" $threshold


## pass all arguments (except the first one) to the python program
#cd source
#python reweighting.py "${@:2}"
#
## tar up the output directory
#cd ..
#tar -zcvf output_"$1".tar.gz -C output/ .
#
## clean up all subdirectories
#rm -rf */
## output.tar.gz should be returned automatically
