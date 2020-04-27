import numpy as np
import itertools 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from Bio import SeqIO

AMINO_ACIDS = np.array([aa for aa in "RKDEQNHSTCYWAILMFVPG-"], "S1")
# AAs = AMINO_ACIDS[:-1] # drop the gap character
AAs = AMINO_ACIDS # idk why Sameer dropped the gap character so I'm not doing that
AAs_string = AAs.tostring().decode("ascii")
AA_L = AAs.size # alphabet size
AA_map = {a:idx for idx,a in enumerate(AAs)} # map each amino acid to an index
# same map as above but with ascii indices
AA_map_str = {a:idx for idx, a in enumerate(AAs_string)}

def get_msa_from_fasta(fasta_filename, size_limit=None, 
                            as_numpy=True):
    """Reads a fasta file and returns an MSA

    Takes a fasta filename and reads it with SeqIO and converts to a numpy
    byte array. This function tries to be fast and keep the data in the
    simplest representation posible. 

    Args:
        fasta_filename  : Filename of fasta file to read
        size_limit      : Return upto size_limit sequences
        as_numpy        : return numpy byte array instead of list of seqs

    Returns:
        A list of sequences (if as_numpy is False)
        A numpy byte array of dtpye S1 which represents the MSA. Each
        sequence is in its own row. (if as_numpy is True)
    """
    seq_io_gen = SeqIO.parse(fasta_filename, "fasta") # generator of sequences
    # Read only size_limit elements of the generator
    # if size_limit is None then we will read in everything
    seq_io_gen_slice = itertools.islice(seq_io_gen, size_limit) 
    seqs = (seq.seq.upper() for seq in seq_io_gen_slice)
    ret = None
    if as_numpy:
        # convert to lists of lists for easy numpy conversion to 2D array
        ret = np.array([list(str(s)) for s in seqs], dtype="|S1")
    else:
        ret = list(seqs)
    return ret


class MSADataset(Dataset):
    '''Reads an MSA and converts to pytorch dataset'''

    def __init__(self, msa_file, size_limit=None, weights=None, transform=None, filterX=False):
        self.raw_data = self.get_raw_data(msa_file, size_limit)
        self.transform = transform
        self.AA_enc = self.get_encoding_dict()

        N = self.__len__()
        if weights is None:
            self.weights = np.ones(N, dtype=np.float);
        else:
            self.weights = np.array(weights).astype(np.float).squeeze()
        assert(self.weights.shape[0]==N)
        if filterX:
            self.filterX()


    #TODO: I'm not sure if this method actually works or not
    def filterX(self):
        print('WARNING: filterX method might be wrong')
        ''' Filters out the proteins and weights for proteins with an X amino acid'''
        g = ((x,w) for (x,w) in zip(self.raw_data, self.weights) if x.find('X'))
        [self.raw_data, self.weights] = list(zip(*g))

    def get_raw_data(self, msa_file, size_limit):
        return get_msa_from_fasta(msa_file, size_limit=size_limit, as_numpy=False)

    def get_encoding_dict(self):
        return AA_map_str.copy()

    def __len__(self):
        return len(self.raw_data)

    def conver_protein_to_tensor(self, prot_str):
        # Integer representation of the protein
        prot_int = [self.AA_enc[p] for p in prot_str]
        sample = torch.LongTensor(prot_int)
        if self.transform:
            sample = self.transform(sample)
        return (sample)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.conver_protein_to_tensor(self.raw_data[idx]), self.weights[idx])
        return sample

class OneHotTransform:
    
    def __init__(self, num_labels, to_float=True):
        self.num_labels = num_labels
        self.to_float = to_float

    def __call__(self, sample):
        ret = F.one_hot(sample, self.num_labels)
        if self.to_float:
            ret = ret.float()
        ret = ret.flatten()
        return ret

