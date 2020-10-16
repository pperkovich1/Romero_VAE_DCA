## Scripts to launch jobs on CHTC

### First time setup
* Install pyyaml on CHTC so that the scripts can read your config files
  ```pip3 install --user pyyaml```

### Running scripts on dataset
Checkout this repo at CHTC. 
```shell

# Clone this repository
git clone https://github.com/RomeroLab/VAEs

# Copy your aligned fasta file to the sequence_sets directory
# Compress it with gzip if it is large
cp my_aligned_msa.fasta sequence_sets/

# edit config file "config.yaml" to set the right aligned_msa
# fasta file. Use nano/emacs/vi or your favourite editor

# change to the run directory
cd run

# clean the output directory if there is anything left over from previous runs
make deepclean 
# run makefile to submit your job and msa to 
make CHTC_MAKE_COMMAND=reweight_and_run chtc_run
```

### Running multiple configs

```shell

# Edit your config.yaml files using vi/nano etc
# Make sure that each config file has a different model name
# so that they do not overwrite each other
# Let us assume they are labeled config_1.yaml, config_2.yaml etc
# and they live in the root directory of this repository

cd run
# submit all configs to CHTC nodes
./submit_all_configs.sh reweight_and_run ../config_*.yaml
```

### Expected Run Times for Reweighting
This is the amount of time re-weighting took on a few
alignment datasets. This is expected to vary considerably by the type of processor 

| Number of sequences | Sequence Length | CHTC (gpu) | CHTC (cpu) | Group Server (CPU) |
| ---   | ---- | --------- |  ---      | ---------|
| ~15k  | ~550 | 0.23 min  |  1.96 min | 0.26 min |
| ~55k  | ~550 | 1.71 min  | 93.32 min |  - min   |
| ~127k | ~200 | 2.18 min    | - min     |  ~30 min  |

### Note for Biochem Cluster (BCC)
All of these scripts can be used on the BCC without any
modification. However, since these scripts use a docker
container and BCC cannot currently use the GPU under docker,
none of the scripts can use the GPU. 
