import pathlib

import numpy as np
import itertools 
import gzip

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from Bio import Seq, SeqIO

AMINO_ACIDS = np.array([aa for aa in "RKDEQNHSTCYWAILMFVPG-"], "S1")
# AAs = AMINO_ACIDS[:-1] # drop the gap character
AAs = AMINO_ACIDS # idk why Sameer dropped the gap character so I'm not doing that
AAs_string = AAs.tostring().decode("ascii")
AA_L = AAs.size # alphabet size
AA_map = {a:idx for idx,a in enumerate(AAs)} # map each amino acid to an index
# same map as above but with ascii indices
AA_map_str = {a:idx for idx, a in enumerate(AAs_string)}

def get_msa_from_fasta_iter(fasta_filename, size_limit=None):
    """Reads a fasta file and returns a string iterator  

    Args:
        fasta_filename  : Filename or filehandle of fasta file to read

    """
    seq_io_gen = SeqIO.parse(fasta_filename, "fasta") # generator of sequences
    # Read only size_limit elements of the generator
    # if size_limit is None then we will read in everything
    seq_io_gen_slice = itertools.islice(seq_io_gen, size_limit) 
    # Here we can return a generator expression because SeqIO.parse
    # is handling the file handle
    return (seq.seq.upper() for seq in seq_io_gen_slice)
 

def get_msa_from_fasta(fasta_filename, size_limit=None, 
                            as_numpy=True, as_iter=False):
    """Reads a fasta file and returns an MSA

    Takes a fasta filename and reads it with SeqIO and converts to a numpy
    byte array. This function tries to be fast and keep the data in the
    simplest representation posible. 

    Args:
        fasta_filename  : Filename or filehandle of fasta file to read
        size_limit      : Return upto size_limit sequences
        as_numpy        : return numpy byte array instead of list of seqs

    Returns:
        if as_iter is True:
            An iterator of sequences in raw string format
        if as_numpy is False:
            A list of sequences in Bio.Seq format
        if as_numpy is True:
            A numpy byte array of dtpye S1 which represents the MSA. 
            The first axis is the sequence number. The second axis is the
            residue number. 
    """
    seqs = get_msa_from_fasta_iter(fasta_filename, size_limit)
    ret = None
    if as_iter:
        ret = seqs
    elif as_numpy:
        # convert to lists of lists for easy numpy conversion to 2D array
        ret = np.array([list(str(s)) for s in seqs], dtype="|S1")
    else:
        ret = list(seqs)
    return ret

def get_msa_from_aln_iter(aln_filename, size_limit):
    """Reads a (plain text) aln file (can be gzipped also) and iterate
        over string sequences 

    Args:
        aln_filename    : Filename or filename of ALN file to read
        size_limit      : Return upto size_limit sequences
    """
    opener = open
    if aln_filename.endswith(".gz"):
        opener = gzip.open
    with opener(aln_filename, "rt") as fh:
        seq_io_gen = (line.strip() for line in fh)
        # Read only size_limit elements of the generator
        # if size_limit is None then we will read in everything
        seq_io_gen_slice = itertools.islice(seq_io_gen, size_limit) 
        # We need the yield from statement below so that the file
        # handle is kept open as long as we are reading from it
        # returning a generator expression would close fh
        yield from (seq.upper() for seq in seq_io_gen_slice)

def get_msa_from_aln(aln_filename, size_limit=None, 
                            as_numpy=True, as_iter=False):
    """Reads a (plain text) aln file (can be gzipped also) and returns an MSA

    Takes a simple text file (ALN) which has one sequence per line. Returns 
    and MSA as a numpy array or as a list of Bio.Seq sequences.

    Args:
        aln_filename    : Filename or filename of ALN file to read
        size_limit      : Return upto size_limit sequences
        as_numpy        : return numpy byte array instead of list of seqs
        as_iter         : return the raw string iterator instead 

    Returns:
        if as_iter is True:
            A raw string iterator
        if as_numpy is False:
            A list of sequences in Bio.Seq format
        if as_numpy is True:
            A numpy byte array of dtpye S1 which represents the MSA. 
            The first axis is the sequence number. The second axis is the
            residue number. 
    """
    seqs = get_msa_from_aln_iter(aln_filename, size_limit)
    ret = None
    if as_iter: # return the raw iterator
        ret = seqs
    elif as_numpy:
        # convert to lists of lists for easy numpy conversion to 2D array
        ret = np.array([list(s) for s in seqs], dtype="|S1")
    else:
        ret = [Seq.Seq(s) for s in seqs]
    return ret


def get_msa_from_file(msa_file, size_limit=None, as_numpy=True, as_iter=False):
    """
        Read in the filename and call the right function to read in the MSA
        by looking at the extension
    """
    suffixes = pathlib.Path(msa_file).suffixes
    suffix = suffixes[-1]
    if len(suffixes) > 1 and suffixes[-1] == ".gz":
        suffix = suffixes[-2] # handle .fasta.gz
    if suffix == ".fasta" or suffix == ".a2m":
        file_reader_func = get_msa_from_fasta
    elif suffix == ".txt" or suffix == ".text" or suffix == ".aln":
        file_reader_func = get_msa_from_aln
    else:
        err_str = f"Input MSA must have format FASTA/A2M or TEXT/ALN.\n" \
                  f"The file extension must be one of " \
                  f".fasta/.a2m/.txt/.text/.aln to reflect that. Or " \
                  f".fasta.gz/.a2m.gz/.txt.gz/.text.gz/.aln.gz if compressed." \
                  f"Found extension {suffix}"
        raise ValueError(err_str)
    return file_reader_func(msa_file, size_limit=size_limit, as_numpy=as_numpy,
            as_iter=as_iter)


class MSADataset(Dataset):
    '''Reads an MSA and converts to pytorch dataset'''

    def __init__(self, msa_file, size_limit=None, weights=None, transform=None):
        self.raw_data = self.get_raw_data(msa_file, size_limit)
        self.transform = transform
        self.AA_enc = self.get_encoding_dict()

        N = self.__len__()
        if weights is None:
            self.weights = np.ones(N, dtype=np.float);
        else:
            self.weights = np.array(weights).astype(np.float).squeeze()
        assert(self.weights.shape[0]==N)

    def get_raw_data(self, msa_file, size_limit):
        return get_msa_from_file(msa_file, size_limit=size_limit, as_numpy=False)

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
    
    def __init__(self, num_labels, to_float=True, flatten=True):
        self.num_labels = num_labels
        self.to_float = to_float
        self.flatten = flatten

    def __call__(self, sample):
        ret = F.one_hot(sample, self.num_labels)
        if self.to_float:
            ret = ret.float()
        if self.flatten:
            ret = ret.flatten()
        return ret

