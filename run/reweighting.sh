#!/bin/bash

# Shell script to launch reweighting script

cd source || { echo "Error: Cannot change to source directory" ; exit 1; }
python reweighting.py "$@"
cd ..
