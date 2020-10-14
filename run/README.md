## Scripts to launch jobs on CHTC

### First time setup
* Install pyyaml on CHTC so that the scripts can read your config files
  ```pip3 install --user pyyaml```

### Reweighting dataset
Checkout this repo at CHTC. 
```shell

# Clone this repository
git clone https://github.com/RomeroLab/VAEs

# Copy your aligned fasta file to the sequence_sets directory
cp my_aligned_msa.fasta sequence_sets/

# change to the run directory
cd run

# edit the pytorch_reweighting.sub file to change the input
# fasta file. Use nano/emacs/vi or your favourite editor

# Replace cmx_aligned_blank_90 with my_aligned_msa in the
# second last line. Also set the threshold param (default 0.8)

# run makefile
make chtc_submit_reweighting
```

### Training
Checkout this repo at CHTC.
```shell

# Clone this repository
git clone https://github.com/RomeroLab/VAEs

# Copy your aligned fasa file to the sequence_sets directory
cp mv_aligned_mas.fasta sequence_sets/

# Change to the configs directory
cd configs

# Create a config file stating your fasta file name and model parameters
# See example for a template.
# If you are testing multiple parameters, you can edit 'make_configs.py'

# Change to the run directory
cd ../run

# If you didnt run make_config.py, create a .txt file with the names of your config files.
# See example_configs.txt as a template

# If you ran 'make_configs.py', it will have already generated one in the 'config' directory. You need to move it to 'run'
mv ../configs/configs.txt .

# Edit the the last line of chtc_submit_training.sub file
queue config from run/your_configs.txt

# run makefile
make submit_training
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
