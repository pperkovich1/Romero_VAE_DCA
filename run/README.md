## Scripts to launch jobs on CHTC

### Usage instructions 
Checkout this repo at CHTC. 
```shell

# Clone this repository
git clone https://github.com/RomeroLab/VAEs

# Copy your aligned fasta file to the sequence_sets directory
cp my_aligned_msa.fasta sequence_sets/

# change to the chtc directory
cd chtc

# edit the pytorch_reweighting.sub file to change the input
# fasta file. Use nano/emacs/vi or your favourite editor

# Replace cmx_aligned_blank_90 with my_aligned_msa in the
# second last line. Also set the threshold param (default 0.8)

# run makefile
make chtc_submit_reweighting
```

### Expected Run Times for Reweighting
This is the amount of time re-weighting took on a few
alignment datasets. This is expected to vary considerably by the type of processor 

| Number of sequences | Sequence Length | CHTC (gpu) | CHTC (cpu) | Group Server (CPU) |
| ---   | ---- | --------- |  ---      | ---------|
| ~15k  | ~550 | 0.23 min  |  1.96 min | 0.26 min |
| ~55k  | ~550 | 1.71 min  | 93.32 min |  - min   |
| ~127k | ~200 |  - min    | - min     |  ~30 min  |

### Note for Biochem Cluster (BCC)
All of these scripts can be used on the BCC without any
modification. However, since these scripts use a docker
container and BCC cannot currently use the GPU under docker,
none of the scripts can use the GPU. 
