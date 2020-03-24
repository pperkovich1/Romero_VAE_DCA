'''
Modified from Sameer's VAE code
'''

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from Bio import SeqIO
import config

class MSADataset(Dataset):
    '''Reads an MSA and converts to pytorch dataset'''

    def __init__(self, msa_file, size_limit = None, weights=None, transform=None, filterX=False):
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
        f =  SeqIO.parse(open(msa_file), 'fasta')
        f = list(f)[:size_limit]
        f = [x.seq for x in f]
        return f

    def get_encoding_dict(self):
        return config.AA_map_str.copy()

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
