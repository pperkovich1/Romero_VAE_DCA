# Sequence Datasets

## cmx\_sequences.zip

- Contains ~15k amino acid sequences from Uniprot 90.

Unzipped, this file contains 2 FASTA files:

- cmx\_alignment.fasta
- cmx\_unaligned.fasta

Here, the `unaligned.fasta` file contains the same sequences as
`cmx_alignment.fasta`, but aligned.


## cmx\_aligned\_blank\_90.fasta

- Contains ~15k amino acid sequences from Uniprot 90 (90% sequence similarity
  cut-off)
- Identical to `cmx_alignment.fasta`, but removed any positions with >90%
  blanks.

## cmx\_wildtype.fasta

- Just picked a random sequence to be wildtype. (TODO: replace with an actual WT if once exists)
