''' Author: Juan R. Diaz Rodriguez
last updated: 2019-05-28 JRD
'''
from torch import nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os


aa2num = {'G':0,'A':1,'B':2,'D':2,'Z':3,'E':3,'K':4,'R':5,
             'H':6,'V':7,'I':8,'S':9,'T':10,'Y':11,
             'N':12,'Q':13,'W':14,'F':15,'P':16,'M':17,
             'L':18,'C':19,'.':20, 'X':20,'-':20}

num2aa = {val:key for key, val in aa2num.items()}

class SequenceDataset(Dataset):
    '''Data loader class for sequences.
    '''
    
    def __init__(self, dataset_file, where=os.getcwd(), transform=None):
        '''
        dataset_file (string): path to the dataset csv.
        transform (callable, optional): Optional transform to be applied.
        '''
        self.sequences = pd.read_csv(dataset_file)
        self.transform = transform
        self.where = where

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx): 
        seq_path = os.path.join(self.where, self.sequences.iloc[idx, 1])
        seq = read_sequence(seq_path)
        if self.transform:
            seq = transform(seq)
        return torch.tensor(np.array(seq, dtype=np.float)).float()


def read_sequence(sequence_file, fmt='fasta'):
    rec = SeqIO.parse(open(sequence_file), fmt)
    rec = next(rec)
    return list(rec.seq)

def filter_hmmer_data(hmmerResults, coverage, identity):
    stats = pd.read_csv('stats.csv')
    stats['coverage'] = pd.to_numeric(stats['coverage'])
    stats['identity'] = pd.to_numeric(stats['identity'])
    stats = stats.sort_values(['coverage', 'identity'], ascending=False)
    stats = stats.drop_duplicates('id') #keeps first element found.

    filtered = stats[(stats['coverage'] >= coverage) & (stats['identity'] >= identity)]
    f = open('summary_'+str(coverage)+'_'+str(identity)+'.txt', 'w')
    f.write('Number of unique sequences: '+str(len(filtered['id']))+'\n')
    f.write('Average identity: '+str(np.average(filtered['identity']))+' Std:'+str(np.std(filtered['identity']))+'\n')
    f.write('Average coverage: '+str(np.average(filtered['coverage']))+' Std:'+str(np.std(filtered['coverage']))+'\n')

    msa = list(zip(*filtered['aligned_sequence']))
    msa = list(zip(*[row for row in msa if (len(set(row)) > 1) or ('-' not in set(row))]))
    seq_records = [SeqRecord(Seq(''.join(seq)), id=name) for seq, name in zip(msa, filtered['id'])]
    SeqIO.write(seq_records, open('cmx_filtered_'+str(coverage)+'_'+str(identity)+'.fasta', 'w'), format='fasta')
    return None

def create_seq_folder(dirname, allseqfile, msa,summary_name='summary'):
    os.mkdir(dirname)
    recs = SeqIO.parse(open(allseqfile), 'fasta')
    ids = pd.DataFrame(columns=['Sequence'])
    for i in len(msa):
        idx = str(i)+'.fasta'
        ids.append({'Sequence': idx})
        SeqIO.write([SeqRecord(Seq(msa[i]))], open(os.path.join(dirname,idx), 'w'), 'fasta')
    ids.to_csv(os.path.join(dirname, summary_name))
    return None

def seq2im(seq, flatten=True, keep_pos=None, unique_aa=None):
    '''Converts a sequence into a binary matrix representation.
        seq (list): sequence to convert.
        flatten (bool): If set to true the function returns a flattened 1D
                        array f length len(seq)*21
    '''
    if flatten:
        length = np.sum([len(unique_aa[i]) for i in range(len(unique_aa)) if i in keep_pos]) #total number of bits
        im = np.array([])
        for pos in keep_pos:
            aa = unique_aa[pos]
            bits = np.zeros(len(aa), dtype=np.float)
            bits[aa.index(seq[pos])] = 1
            im = np.append(im, bits)
        return im 
    else:
        im = np.zeros(len(seq),21)
        for i in range(im):
            im[i][aa2num[seq[i]]] = 1
        return im #return matrix form

def seq2bit_exclude_loops(seq, keep_pos, unique_aa):
    length = 0
    for i in range(len(unique_aa)):
        if i in keep_pos:
            length += len(unique_aa[i])
    im = np.array([])
    #encode and build vector.
    for pos in keep_pos:
        aa = unique_aa[pos]
        bits = np.zeros(len(aa), dtype=np.int8)
        bits[aa.index(seq[pos])] = 1
        im = np.append(im, bits)
    return im 

def bit2seq_exclude_loops(bitseq, seq, keep_pos, unique_aa):
    '''Reconstructs a bit-encoded sequence back into a protein sequence.
    '''
    seq_recon = []
    bitix = 0
    for i in range(len(seq)):
        if i in list(keep_pos):
            bitix_next = bitix+len(unique_aa[i])
            aa = unique_aa[i][np.where(bitseq[bitix:bitix_next] == 1.0)[0][0]]
            bitix = bitix_next
            seq_recon.append(aa)
        else:
            seq_recon.append(seq[i])
    return seq_recon

def im2seq(seqim):
    '''Converts binary representation of a sequence back to readable format.
    '''
    if len(np.shape(seqim)) > 1: # unflattened case
        return [num2aa[np.where(pos == 1)[0][0]] for pos in seqim]
    else: #flatenned case. There's probably a better way to do this block.
        s = seqim.reshape(len(seqim)/21, 21) 
        return [num2aa[np.where(pos == 1)[0][0]] for pos in s]

def seq2bit(seq, allowed, keep_pos=None):
    '''Converts a sequence into a bit encoded, one dimensional
    vector where each position is encoded by the least number of bits possible.
    
    Parameters:
        seq: list - list of strings that make up the protein sequence.
        allowed: list - list of tuples. Each tuple contains all possible
        amino acids found for that position.
        keep_pos(optional): list - index of positions to use when making a bit
        encoded sequence. This allows for pre-filtering positions due to blank
        percertage or any other metric.
    '''
    if keep_pos == None:
        bit_seq = np.zeros(len([len(t) for t in allowed]),dtype=np.int8)
        ix = 0
        for i,aa_list in enumerate(allowed):
            aa_num = len(aa_list)
            bits = np.zeros(aa_num, dtype=np.int8)
            bits[aa_list.index(seq[i])] = 1
            bit_seq[ix:ix+aa_num] = bits
            ix += aa_num
        return bit_seq
    else:
        return seq2bit_exclude_loops(seq, keep_pos, allowed)
    return None

def bit2seq(bit_seq, allowed, keep_pos=None):
    '''Converts a bit-encoded sequence back to its original protein sequence.
    Uses allowed and keep_pos(if provided) to 
    
    Parameters:
        bit_seq: list - bit encoded sequence.
        allowed: list - list of tuples. Each tuple contains all possible
        amino acids found for that position.
        keep_pos(optional): list - index of positions to use when making a bit
        encoded sequence. This allows for pre-filtering positions due to blank
        percertage or any other metric.
    '''
    if keep_pos == None:
        seq_recon = []
        bitix = 0
        for i, aa_list in enumerate(allowed):
            aa_num = len(aa_list)
            bitix_next = bitix+aa_num
            aa = aa_list[np.where(bit_seq[bitix:bitix_next] == 1)[0][0]]
            bitix = bitix_next
            seq_recon.append(aa)
        return seq_recon  

    else:
        return bit2seq_exclude_loops(bit_seq, keep_pos, allowed)
    return None

def msa2im(msa, classify=False):
    '''Converts each sequence into a NxM binary matrix where N is the aa number
    and M is the number of positions in the sequence. each row corresponds to
    the position and each column represents an AA. See aa2num dict defined
    above to know which column corresponds to which column.
    
    Adds the classifier to each as functional
    '''
    
    data_set = np.zeros((len(msa),len(msa[0]),21))
    for i, seq in enumerate(msa):
        for j, aa in enumerate(seq):
            data_set[i][j][aa2num[aa.upper()]] = 1
        data_set[i] = data_set[i]
        
    if classify:
        return [(d, 1) for d in data_set]
    
    return [[d] for d in data_set]

def random_seq(seq_len, aa_freq=None):
    '''Creates a random sequence as a binary image matrix. Columns are amino
    acids, rows are positions. returns the (sequence matrix, classifier) tuple.
    '''
    seq = np.zeros((seq_len, 21))
    for i in range(len(seq)):
        seq[i][np.random.randint(0, 20)] = 1
    return (torch.tensor(seq), 0)

def coverage(seq1, seq2):
    '''Calculates the coverage fraction of seq2 on seq1.
    '''
    if '.' in seq1+seq2:
        seq1 = list(''.join(seq1).replace('.','-'))
        seq2 = list(''.join(seq2).replace('.','-'))

    denom = len(seq1) - seq1.count('-')
    c = 0
    for aa1,aa2 in zip(seq1,seq2):
        if aa1 == '-':
            pass
        if aa2 != '-':
            c += 1
    return c/denom

def identity(seq1, seq2):
    '''Calculates pairwise identity between the two sequences.
    Assumes sequences are aligned.
    '''
    t = len(seq1)
    count = 0
    for s1,s2 in zip(seq1, seq2):
        if len(set([s1, s2])) == 1:
            count += 1
    return count/t

def trim_msa(msa):
    '''Trims an MSA of empty(gap-only) columns.
    Assumes MSA is in format:
    [[seq1],
    [seq2],
    ...,
    [seqN]]
    where [seq] is a list of characters.
    '''
    msa_seqs = list(zip(*msa))
    msa_seqs_trimmed = []
    for i,col in enumerate(msa_seqs):
        types = set(col)
        if len(types) == 1 and ('-' in types):
            pass
        else:
            msa_seqs_trimmed.append(col)
    return list(zip(*msa_seqs_trimmed))


def bin_seq(seq, keep_pos, unique_aa):
    '''Force one hot-encoding of sequence made up of floats.
    The encoding corresponds to each position for
    the reduced dimensionality encoded sequence.
    '''
    seq_recon = np.array([], dtype=np.int)
    bitix = 0
    for pos in keep_pos:
        bitix_next = bitix+len(unique_aa[keep_pos])
        bin_v = np.copy(seq[bitix:bitix_next])
        bin_v = np.array(bin_v >= max(bin_v), dtype=np.int)
        bitix = bitix_next
        seq_recon = np.append(seq_recon, bin_v)
    return seq_recon

def binarize_image(v):
    bin_v = np.zeros(v.shape)
    for i,row in enumerate(v):
        bin_v[i] = v[i] >= max(row)
    return np.array(bin_v).astype(np.int8)

def get_cutoffs(msa, blank_cutoff=.7):
    '''
    Assumes: msa provided in column = seq, row = pos
    '''
    num_seqs = len(msa[0])
    num_pos = len(msa)
    blank_freq = np.zeros(num_pos)
    unique_aa_num = np.zeros(num_pos)
    for i, pos in enumerate(msa):
        unique_aa_num[i] = len(set(pos))
        blank_freq[i] = pos.count('-')
    blank_freq = blank_freq/num_seqs
    #get positions to keep
    blank_keep = np.nonzero(blank_freq < blank_cutoff)[0] # get indices that meet our criteria
    unique_aa = [list(set(pos)) for pos in msa] #list of [[aa's]]
    return blank_freq, unique_aa_num, blank_keep, unique_aa #return all for future reference and backtracking

def generate_limits(unique_aa_num, keep=[]):
    lims = []
    i = 0
    for pos,aa_num in enumerate(unique_aa_num):
        if pos not in keep:
            pass
        else:
            lims.append([i,i+aa_num])
            i += aa_num
    return lims
    

def binarize_tensor(tens, lims):
    ''' Binarizes tensor using the limits that excode each amino acid.
    See get_cutoffs and generate_limits for details on how the limits are generated. 
    '''
    for lim1,lim2 in lims:
        tens[lim1:lim2] = tens[lim1:lim2] >= torch.max(tens[lim1:lim2]).item()
    return tens.float()
    
def tensor_pairwise_identity(t1, t2, lims, keep=[]):
    # binarize, cast and copy without grad.
    t1_clone = binarize_tensor(t1, lims).clone().detach()
    t2_clone = binarize_tensor(t2, lims).clone().detach()
    if len(keep) == 0: # if all should count.
        product = t1_clone.dot(t2_clone).item()
        return product/len(lims) #largest number in lims
#    else:
#        t1_clone = 
#        t2_clone = 
#        product = t1_clone.dot(t2_clone).item()
#        return product/len(keep)
    
    


