#!/bin/bash

# Usage:
#   To run several configs with one command place all the configs somewhere
#   Then use bash wildcards to run this script. The first
#   argument is the makefile command from ../Makefile which will be run 
#   All the next arguments are config files that need to be run with this command
# 
#   Example 1: (single config file)
#   ./submit_all_configs.sh reweighting ../config.yaml
#   Example 2: (multiple config files)
#    ./submit_all_configs.sh reweight_and_run ../config*.yaml
#

if [ $# -lt 2 ] 
then
    echo "$# arguments supplied. Must supply atleast two."
    echo "Usage: $0 make_file_command config [configs...]"
    echo "Example: $0 reweight_and_run ../config.yaml ../config_some.yaml"
    exit 1
fi

CHTC_MAKE_COMMAND=$1
for config in "${@:2}"
do
    make CHTC_MAKE_COMMAND=${CHTC_MAKE_COMMAND} \
                CONFIG="${config##*/}" \
                CONFIG_DIR="${config%/*}" \
                chtc_run 
done
