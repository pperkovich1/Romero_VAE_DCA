"""
Created on Tue Sep  3 10:22:49 2019

@author: jrdia

This script parses and gathers info on identity and coverage from the jackhmmer
result alignment. It saves a CSV with these stats for future reference and
processing.
"""

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import argparse
import pandas as pd



#TODO
#finish arg parser
#modularize and document functions
#check for optimization opportunities.

#hmmer_file='results.fasta', fmt='fasta', no_ss=True

parser = argparse.ArgumentParser(description='Reads and alignment file and\
                                 provides preliminary metrics such as pairwise\
                                 identity and coverage to a query sequence.')
parser.add_argument('-hmmer_file',
                    type=str,
                    default='alignment.fasta',
                    help='hmmer output file. Assumes hmm3_tab format if not specified')

parser.add_argument('-no_ss',
                    action='store_true',
                    help='Option to remove substrings when parsing. Default False.')
parser.add_argument('-out',
                    type=str,
                    default='msa_stats.csv',
                    help='Name of the csv file produced by the script. Default: msa_stats.csv')
parser.add_argument('-fmt',
                    type=str.lower, #forces lower case string on argument
                    default='fasta',
                    help='Option to remove substrings when parsing. Default FASTA.')


args = parser.parse_args()

##########






def identity(seq1,seq2):
    """Calculates identity of two pre-aligned sequences.
    """
    ct = 0
    match = 0
    for aa1,aa2 in zip(seq1, seq2):
        if aa1 == '-' or aa2 == '-':
            pass
        else:
            ct +=1
            match += int(aa1 == aa2)
    return match/ct

def coverage(query, hit):
    """
    Calculates coverage of between two sequences.
    """
    ct = 0
    match = 0
    for aa1,aa2 in zip(query, hit):
        if aa1 == '-':
            pass
        else:
            ct +=1
            match += int(aa2 != '-')
    return match/ct


def filter_pos_by_blank(msa, cutoff=0.9):
    """
    Removes positions from an msa that are over the specified blank cutoff.
    """
    msa_pos = list((zip(*msa))) #transpose the msa.
    l = len(msa_pos[0])
    msa_blanks_cutoff = []
    add_pos = msa_blanks_cutoff.append #makes appending a little bit faster when looping
    for i,pos in enumerate(msa_pos):
        if pos.count('-')/l < cutoff: # check no. of blanks at pos
            add_pos(pos)
    return list(zip(*msa_blanks_cutoff))

def remove_blanks_from_seq(seq):
    """ Removes gaps, or blanks, from sequences in an MSA.
    
    Parameters:
        seq - str: sequence to remove blanks from.
    Assumes:
        blanks are represented by the dash (-) character
    """
    return seq.replace('-','')

def remove_blanks_from_rec(seq_record):
    """ Removes gaps, or blanks, from a SeqRecord object.
    
    Parameters:
        seq_record - Bio.SeqRecord: sequence record to remove blanks from.
    Returns:
        seq_record_no_blanks: seq_record with gaps removed from the sequence.
    Assumes:
        blanks are represented by the dash (-) character
    """
    return SeqRecord(Seq(str(seq_record.seq).replace('-',''), id=seq_record.id))

def remove_substrings_from_alignment(alignment):
    """Removes substrings from a list of SeqRecord objects.

    Parameters:
        alignment - list: list of SeqRecord objects
    Returns:
        list of SeqRecord objects without substrings.
    """
    keep = []
    for i, rec1 in enumerate(alignment):
            seq1 = remove_blanks_from_seq(str(rec1.seq))
            for j, rec2 in enumerate(alignment):
                if i == j:
                    pass
                else:
                    seq2 = remove_blanks_from_seq(str(rec2.seq))
                    # checks for substring
                    if seq1 not in seq2:
                        keep.append(i)
    return [alignment[i] for i in keep]


def get_msa_stats(hmmer_file='alignment.fasta', fmt='fasta', no_ss=True, out='msa_stats.csv'):
    """ Calculate coverage and identity for an MSA. Initially made to process
    hmmer files, it can be used to extract coverage and identity information
    from any MSA.
    
    
    Assumes:
        First sequence in the MSA is the query sequence.
        MSA file in in a standard file format supported by the Biopython package.
    """
    alignment = list(SeqIO.parse(open(hmmer_file), fmt))
    l = len(alignment)
    keep = [0]

    if no_ss:
        alignment = remove_substrings_from_alignment(alignment)
    print('Substrings removed:', str(l - len(alignment)))
    #add query sequence to the stats file
    alignment_stats = [{'id':alignment[0].id,
                           'identity':1.,
                           'coverage':1.0,
                           'seq_aln':''.join(alignment[0].seq)}]
    # get identity and coverage for each sequence in the alignment
    query_seq_list = list(str(alignment[0].seq))
    for seq_record in alignment[1:]: #don't include the query sequence
        hit_rec_list = list(str(seq_record.seq))
        alignment_stats.append({'id':seq_record.id,
                                  'identity':identity(query_seq_list, hit_rec_list),
                                  'coverage':coverage(query_seq_list, hit_rec_list),
                                  'seq_aln':''.join(seq_record.seq)})
    l2 = len(alignment_stats)
    alignment_stats = pd.DataFrame(alignment_stats)
    print('Results written to:', out)
    print("Average identity:", str(alignment_stats['identity'].mean()))
    print("Average Coverage:", str(alignment_stats['coverage'].mean()))

    #write data frame into csv
    alignment_stats.to_csv(out)

    print("Total number of sequences written:", str(l))
    return None
if __name__ == '__main__':
    get_msa_stats(**vars(args))
