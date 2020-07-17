"""Script to reweight sequences from multiple MSAs according to evolution

  Input transition matrix: This needs to be a pandas dataframe of the 
                           mutational biases each round.
                           ini           A         C         G         T
                           trans                                        
                           A      0.991548  0.000785  0.001805  0.002944
                           C      0.000740  0.997419  0.000158  0.003917
                           G      0.004625  0.000146  0.997154  0.000604
                           T      0.003086  0.001650  0.000883  0.992535
                           (WARNING): The columns are the initial nucleotides 
                                      and the rows are the nucleotides they
                                      transition to.  
                                      p(A -> C) = transition_mat['C', 'A']

  MSAs:                    These need to be nucleotide based MSAs

  Output weights:          (WARNING): These end up being very large. Floating
                                      point array of size (n, L, q)
  Output MSA:              All the nucleotide MSAs will be concatenated into
                           one giant Amino Acid MSA. The order of sequences
                           will match up with the output weights

  Typical usage example: (from the command line)
	python evo_weight_tools.py \
		--nt_transition_file pd_transition_matrix.pkl \
		--output_weights ../working/evo_weights.npy \
                --msa_prev  msa_round_1.txt.gz
		--msa  msa2_round_2.txt.gz

"""

import numpy as np
import pandas as pd

import torch
import tqdm # for progress bar

from scipy.special import logsumexp
import Bio

import dataloader

def create_codon_transition_matrix(nucleotide_trans_mat):
    """ Calculate codon transition matrix from nucleotide transition matrix
        
        Assumption: We assume the nucleotide_trans_mat has initial states as
            columns and transitioned states as rows. And the return argument
            shares that assumption. However, this function should work
            transparently. i.e.  if we have the initial states as rows and the
            transitioned states as columns then the same will be true for the
            return argument.  

        Args:
            nucleotide_trans_mat:  a nucleotide transition matrix (4x4) 
                                   (as pandas dataframe)
        Return:
            codon_trans_mat: a codon transition matrix (64x64)(
                             (as pandas dataframe)
    """
    # 6 dim Cartesian product of A,C,T,G
    nucleotide_idxs = np.meshgrid(*([nucleotide_trans_mat.index] * 6))
    n1, n2, n3, n4, n5, n6 = list(map(lambda x: x.flatten(), nucleotide_idxs))
    codons_from = n1 + n2 + n3 # AAA, AAC, ...
    codons_to = n4 + n5 + n6 
    # Calculate transition probability by multiplying individual transitions
    codon_trans = nucleotide_trans_mat.lookup(n4, n1) * \
                    nucleotide_trans_mat.lookup(n5, n2) * \
                    nucleotide_trans_mat.lookup(n6, n3)
    codon_trans = pd.DataFrame(data={'prob':codon_trans, 'to':codons_to,
                                    'from':codons_from})
    # Reshape long to wide to create a lookup table
    codon_trans_mat = codon_trans.pivot(index="to", columns="from", 
                                            values="prob")
    return codon_trans_mat


def get_codon_msa_as_int_array(filename, codon_map):
    seq_iter = dataloader.get_msa_from_file(filename, as_iter=True)

    def codon_seq_to_int_list(seq): 
        return [codon_map[seq[3*i:(3*i+3)]] for i in range(len(seq)//3)]
    
    return np.array([codon_seq_to_int_list(seq) for seq in seq_iter], 
                        dtype=np.uint8) 


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--msa_prev", type=str, required=True,
                    help="Path to Previous MSA")
    parser.add_argument("-m", "--msa", type=str, required=True,
                    help="Path to Current MSA")
    parser.add_argument("-t", "--nt_transition_file", type=str, required=True,
                    help="Path to Nucleotide Transition Matrix"
                         " (pandas pickle format)")
    parser.add_argument("-o", "--output_weights", type=str, required=True,
                    help="Path to weights output file (numpy archive)")
    args = parser.parse_args()

    start_time = time.time()

    nt_trans_mat = pd.read_pickle(args.transition_file)
    codon_trans_mat = create_codon_transition_matrix(nt_trans_mat)
    #print(codon_trans_mat)

    # create a mapping for non-degenerate codons
    # This will be used for one-hot encoding the sequences
    codon_table = Bio.Data.CodonTable.standard_dna_table
    codon_map = {c:i for i, c in enumerate(
                        sorted(codon_table.forward_table.keys()))}

    msa_prev = get_codon_msa_as_int_array(args.msa_prev, codon_map)
    msa_curr = get_codon_msa_as_int_array(args.msa, codon_map)

    # create indexing for rows and columns of transition matrix
    # in the same order as the codon_map
    codon_rows = np.array([i for i, c in enumerate(codon_trans_mat.index) 
                                if c in codon_map]) 
    codon_cols = np.array([i for i, c in enumerate(codon_trans_mat.columns) 
                                if c in codon_map]) 
    # q = 61 non-dengerate codons
    # L = 186 Residue Length of protein
    # n = number of proteins in msa_prev
    # m = number of proteins in msa_curr

    # subset transition map to non-degenerate codon transition matrix
    # Probabilities are stored as log probability
    # Probability of transition goes from columns to rows
    # Shape (q, q)
    nondeg_codon_trans_mat = np.log(codon_trans_mat.iloc[codon_rows, 
                                        codon_cols].to_numpy())

    # numpy integer array for indexing
    # shape (L,)
    range_prot = np.arange(msa_curr.shape[1], dtype=np.uint8)

    # Probability of transition from previous MSA codons to any other codon
    ptrans = nondeg_codon_trans_mat[:, msa_prev] # shape (L, n, q)
    ptrans = np.moveaxis(ptrans, [1,2], [0,1]) # shape (n, L, q)

    N_curr = msa_curr.shape[0] # number of elements in current MSA
    output_weights = np.zeros((N_curr, ptrans.shape[1], ptrans.shape[2]),
            dtype=np.float) # shape (N_curr, L, q)
    for i in tqdm.trange(N_curr):
        # one protein in the current MSA
        y = msa_curr[i, :] # shape (L,) 
        # probability of transition from msa_prev to y split out by codon position
        ptrans_y = ptrans[:, range_prot, y] # shape (n, L)
        # probability of transition from msa_prev to y aggregated
        ptrans_y_agg = ptrans_y.sum(axis=1) # shape (n, )

        # probability of transition from msa_prev to single mutants of y
        ptrans_y_muts = (ptrans + ptrans_y_agg[:, np.newaxis, np.newaxis] - 
                ptrans_y[:, :, np.newaxis]) # shape (n, L, q)
        # earlier we were adding log probability but now we need to actually add
        # probability. So we use logsumexp instead of sum.
        ptrans_y_muts_agg = logsumexp(ptrans_y_muts, axis=0)
        output_weights[i, :, :] = ptrans_y_muts_agg
        #if (i > 3): break

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    np.save(args.output_weights, output_weights, allow_pickle=False)
 
