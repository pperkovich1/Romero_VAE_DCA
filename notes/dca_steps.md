# Steps to build a Direct Coupling Analysis (DCA) model

We start with a query sequence and build a multiple sequence alignment (MSA) of sequences similar in identity to the query sequence. 
Then we run a DCA model on the the resulting MSA to get main effect and interaction parameters. 


### Building the MSA
1. Create a new empty directory and save a query sequence in `fasta` format there. 
An example query sequence for [mDHFR is here](mDHFR.fasta) and it should not have any gaps or unknown amino acids in it. 
1. Create an MSA by running the the `jackhmmer` program to search for similar sequences in the `uniprot` database. 
The query sequence is automatically saved as the last record in the MSA.
The following commands are run on the group server and take around 30 minutes or so. 
   ```shell
   # create MSA in stockholm format
   jackhmmer -A mDHFR.sto -o mDHFR.out.txt mDHFR.fasta /mnt/scratch/databases/uniprot/uniref90.fasta
   
   # convert stockhom format to aligned fasta (afa) format
   /home/romeroroot/code/hmmer-3.1b2-linux-intel-x86_64/binaries/esl-reformat -u -o mDHFR.afa afa mDHFR.sto 
   ```
1. Filter the columns of the MSA. There are several extra columns in the output and we only want to keep those that correspond to our query sequence.
   ```shell
   # Load up the pytorch-docker conda environment
   conda activate pytorch-docker
   # Run the filtering script
   python3 last_record_filter.py -i mDHFR.afa -o mDHFR_clean.fasta 
   ```
1. Now you should have a clean MSA file. It should have atleast a few thousand sequences in it before you can proceed to the next step.
   
### Building DCA model
Here we use [`plmc`](https://github.com/debbiemarkslab/plmc) from the Marks lab to build a DCA model. 
1. Run `plmc` on the cleaned MSA file and save the model parameters
   ```shell
   ~romeroroot/code/plmc/bin/plmc -o mDHFR_params.bin -c mDHFR_couplings.txt mDHFR_clean.fasta
   ```
2. Use the plmc Matlab script `~romeroroot/code/plmc/scripts/read_params.m ` to read the parameters. 
