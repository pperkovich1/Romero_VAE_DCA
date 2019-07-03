#!/bin/bash

# untar your Python installation
tar -xzf pythonv2.tar.gz
tar -xzf sequences.tar.gz
# make sure the script will use your Python installation, 
# and the working directory as it's home location
export PATH=$(pwd)/miniconda3/bin:$PATH
mkdir home
export HOME=$(pwd)/home
mkdir results
# run your script
# python vae.py 1 100 1e-5 10000 0.01 400 20
# ^^^ slow version          vvv fast version
python vae.py 1 100 1e-3 100 0.1 400 20
