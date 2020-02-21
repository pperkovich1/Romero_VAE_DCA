"""Reweight sequences in an MSA to account for uneven sampling.

Reweight sequences in an MSA to account for unevening sampling and
phylogenetic biases. This script will take an MSA as input (in Fasta or ALN
format) and compute the weights of each sequence. The memory reqiurements and
the computational complexitity for a naive algorithm grow according to the
square of the number of sequences. This isn't a problem for MSAs with ~10k
sequences but quickly becomes a problem with MSAs of size ~100k and above.
The design goal is to be able to use it as a standalone script or as a module
and imported into other python programs.

  Typical usage example:

  weights = compute_weights_from_msa(threshold_distance)
"""

import numpy as np
import torch

# TODO(Sameer): Merge this function with existing code in utils.py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# TODO(Sameer): Merge this function with existing code in utils.py
def get_msa_from_fasta(fasta_filename):
    """Reads a fasta file and returns an MSA

    Takes a fasta filename and reads it with SeqIO and converts to a numpy
    byte array. This function tries to be fast and keep the data in the
    simplest representation posible. 

    Args:
        fasta_filename: Filename of fasta file to read

    Returns:
        A numpy byte array of dtpye S1 which represents the MSA. Each
        sequence is in its own row. 
    """
    # TODO(Sameer): Move SeqIO import to header after merge with utils.py
    from Bio import SeqIO
    seq_io_gen = SeqIO.parse(fasta_filename, "fasta") # generator of sequences
    # convert to lists of lists for easy numpy conversion to 2D array
    seqs = [list(str(seq.seq.upper())) for seq in seq_io_gen]
    return np.array(seqs, dtype="|S1")

def compute_weights_from_msa(msa, threshold, device=device):
    """Computes weights from an msa using threshold to determine neighbors.

    Counts the number of neighbors around each sequence using the threshold
    as a cutoff to determine what a neighbor is. The weight for each sequence
    is the reciprocal of the count of how many neighbors it has.

    Args:
        msa: A numpy array of dtype S1 with 2 dimensions and each sequence as
            a row.
        threshold: A float that determines the cutoff at which a sequence
            with distance similarity to another sequence is considered a
            neighbor. Hamming distance >= 80% would have 0.8 for this value.
        device: pytorch device to store the Tensors. This function can take
            very long to compute on a CPU. For ~100k sequences we estimate
            around 20 minutes. So hopefully device is "cuda:0".

    Returns:
        A numpy 1D array of weights which has the same number of elements as
        the the number of sequences (rows or first dimension) in the msa that
        is passed in. 
    """
    torch.set_grad_enabled(False)
    N, L = msa.shape # number of sequences, length of protein
    msa_int = torch.ByteTensor(msa.view(np.uint8))

    # Compute integer threshold
    distance_to_threshold = threshold 
    epsilon = 1e-6 # to avoid rounding issues (unlikely)
    distance_from_threshold_int = int((1 - distance_to_threshold)*L + epsilon)

    # Create a torch scalar for the threshold
    torch_threshold = torch.ShortTensor(1)
    torch_threshold[0] = distance_from_threshold_int
    torch_threshold = torch_threshold.to(device)

    msa_int = msa_int.to(device) # move to device
    torch_seqs = torch.unbind(msa_int, dim=0) # split into separate sequences

    def count_neighbors(torch_seq):
        """Count the neighbors of torch_seq.
        This is an inline function that uses msa_int and torch_threshold"""
        dist_from_seq = (torch_seq != msa_int).sum(axis=1, dtype=torch.short)
        threshold_count = (dist_from_seq <=
                torch_threshold).sum(dtype=torch.short) 
        return threshold_count

    neighbors_count = torch.stack(tuple(count_neighbors(torch_seq) for
        torch_seq in torch_seqs))

    weights = 1 / neighbors_count.float()
    return weights.data.numpy()



if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfasta", 
                    help="input alignment file in fasta format")
    parser.add_argument("-o", "--outputnpy", 
                    help="output filename for weights file")
    parser.add_argument("-t", "--threshold", default=0.8, type=float,
                    help="Threshold similarity for a sequence to be "
                         "considerd a neighbor")
    args = parser.parse_args()

    msa = get_msa_from_fasta(args.inputfasta)
    start_time = time.time()
    weights = compute_weights_from_msa(msa, threshold=args.threshold)
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    np.save(args.outputnpy, weights, allow_pickle=False)


