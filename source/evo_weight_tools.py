"""Script to reweight sequences from multiple MSAs according to evolution

  Input transition matrix: This needs to be a pandas dataframe of the 
                           mutational biases each round.
                           ini           A         C         G         T
                           trans                                        
                           A      0.991548  0.000785  0.001805  0.002944
                           C      0.000740  0.997419  0.000158  0.003917
                           G      0.004625  0.000146  0.997154  0.000604
                           T      0.003086  0.001650  0.000883  0.992535
                           (WARNING): The columns are the initial nucleotides and the rows
                                      are the nucleotides they transition to.
                                      p(A -> C) = transition_mat['C', 'A']

  MSAs:                    These need to be nucleotide based MSAs

  Output weights:          (WARNING): These end up being very large. Floating point array
                                        of size (n, L, q)
  Output MSA:              All the nucleotide MSAs will be concatenated into one giant
                           Amino Acid MSA. The order of sequences will match up with the
                           output weights

  Typical usage example: (from the command line)
	python evo_weight_tools.py \
		-i pd_transition_matrix.pkl \
		-o ../working/evo_weights.npy
		#-m ../working/evo_msa.txt.gz
                msa_round_1.txt.gz msa2_round_2.txt.gz

"""

import numpy as np
import pandas as pd

import dataloader

def calc_transition_probability(seq_from, seq_to, 
                                    transition_mat):
    return transition_mat.lookup(list(str(seq_to)), list(str(seq_from))).prod()

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



if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('MSAs', metavar='MSAs', type=str, nargs='*',
                                help='MSA files')
    parser.add_argument("-i", "--transition_file", type=str, required=True,
                    help="Path to Nucleotide Transition Matrix"
                         " (pandas pickle format)")
    parser.add_argument("-o", "--output_weights", type=str, required=True,
                    help="Path to weights output file (numpy archive)")
    args = parser.parse_args()

    nt_trans_mat = pd.read_pickle(args.transition_file)
    codon_trans_mat = create_codon_transition_matrix(nt_trans_mat)
    print(codon_trans_mat)

    x = dataloader.get_msa_from_file(args.MSAs[1], as_numpy=False)
    wt = dataloader.get_msa_from_file(args.MSAs[0], as_numpy=False)
    seq1 = x[0]
    seq2 = x[1]
    print(calc_transition_probability(wt[0], seq1, transition_mat))


    print(args.MSAs)


    start_time = time.time()
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    #np.save(config.weights_fullpath, weights, allow_pickle=False)
 