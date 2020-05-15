"""Manipulate MSAs (stats, extract parts etc)

  Typical usage example: (from the command line)
	#python msa_tools.py \
	#	-i ../sequence_sets/cmx_aligned_blank_90.fasta \
	#	-t 0.8 \
	#	-o ../working/cmx_aligned_blank_90_weights.npy
        python msa_tools.py -i ../sequence_sets/cmx_aligned_blank_90.fasta \
                -w ../sequence_sets/cmx_wildtype.fasta                                                                          
    Distance from WT: 62.36%
    Sampling 100k pairs to calculate pairwise distance
    Avg pairwise distance: 63.10%
    Time elapsed: 0.13 min

"""

import functools
import numpy as np

import dataloader

class MSA:
    """ Perform manipulations on MSA 

    Store an MSA as a numpy "|S1" array (most efficient form)
        TODO: Allow torch arrays later

    """

    def __init__(self, msa, wt):
        """
        Args:
            msa    : Numpy array of shape (N, L)  dtype "|S1"
            wt     : Numpy array of shape (1, L) or (L,)  dtype "|S1"
        """
        self.msa = msa
        self.wt = wt
        wt = wt.squeeze()
        if len(wt.shape) != 1:
            raise ValueError("WildType filename must have only one value in it")
        if wt.shape[0] != msa.shape[1]:
            raise ValueError("WildType and MSA files have different length sequences in them")

    @property
    def num_seqs(self):
        return self.msa.shape[0]

    @property
    def seq_length(self):
        return self.msa.shape[1]

    @property
    def avg_dist_from_wt(self):
        return self.calc_avg_dist_from_seq(self.wt)

    @property
    def avg_dist_from_wt_pct(self):
        return self.avg_dist_from_wt  / self.seq_length * 100

    def calc_avg_dist_from_seq(self, seq):
        return (self.msa != seq).sum(axis=1).mean()

    @property
    @functools.lru_cache(1)
    def avg_pairwise_dist(self):
        """ This is not a property because it takes forever to calculate """
        if self.num_seqs > 1000:
            print("Sampling 100k pairs to calculate pairwise distance")
            # sample pairwise indices
            it = (np.random.choice(self.num_seqs, 2, replace=True)
                    for _ in range(100000))
            distances = np.array([(self.msa[x, :] != self.msa[y, :]).sum() for x, y in it])
        else:
            distances = np.array([self.calc_avg_dist_from_seq(self.msa[i, :]) for i in 
                                    range(self.seq_length)])
        return distances.mean()

    @property
    def avg_pairwise_dist_pct(self):
        return self.avg_pairwise_dist / self.seq_length * 100

    @staticmethod
    def create_from_files(msa_filename, wildtype_filename):
        msa = dataloader.get_msa_from_file(args.msa_filename, as_numpy=True)
        wt = dataloader.get_msa_from_file(args.wildtype_filename, as_numpy=True)
        return MSA(msa, wt)


if __name__ == "__main__":
    import time
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--msa_filename",
                    help="input msa in ALN or FASTA format",
                    required=True) 
    parser.add_argument("-w", "--wildtype_filename",
                    help="filename that contains only WildType",
                    required=True) # required for now! TODO: make optional later
    args = parser.parse_args()

    start_time = time.time()

    msa = MSA.create_from_files(args.msa_filename, args.wildtype_filename)

    print(f"Distance from WT: {msa.avg_dist_from_wt_pct:.2f}%")
    print(f"Avg pairwise distance: {msa.avg_pairwise_dist_pct:.2f}%")

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))


