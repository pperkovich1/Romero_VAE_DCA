"""
Created on Tue Mar 10

@author: jrdia
"""

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Reads a csv that contains identity\
                    and coverage metrics. Creates an alignment file after\
                    applying identity and coverage filters provided.')
parser.add_argument('-stats_file',
                    type=str,
                    default='msa_stats.csv',
                    help='CSV file containing alignment metrics.')
parser.add_argument('-max_blank',
                    type=float,
                    default=1.0,
                    help='Maximum number of blanks allower per position. Default: 1.0') # no filtering of blanks
parser.add_argument('-min_cov',
                    type=float,
                    default=0.8,
                    help='Minimum cutoff for filtering by coverage. Default: 0.8')
parser.add_argument('-min_ident',
                    type=float,
                    default=0.1,
                    help='Minimum cutoff for filtering by identity. Default: 0.1')
parser.add_argument('-max_ident',
                    type=float,
                    default=0.9,
                    help='Maximum cutoff for filtering by identity. Default: 0.9')
parser.add_argument('-db_out',
                    type=str,
                    default='final_alignment.fasta',
                    help='Name of the alignment file produced. Default: final_alignment.fasta')
parser.add_argument('-fmt',
                    type=str.lower,
                    default='fasta',
                    help='Format of the alignment file produced. Default: fasta')
args = parser.parse_args()

##########

def filter_pos_by_blank(msa, cutoff=0.9):
    """ Removes positions from an msa that are over the specified blank cutoff.

    Parameters:
        msa - list: list of sequences. each sequence is a list of strings.
        cutoff - float (optional): coverage cutoff
    """
    msa_pos = list((zip(*msa))) #transpose the msa such that rows correspond to positions in the msa.
    l = len(msa_pos[0])
    msa_blanks_cutoff = []
    for pos in msa_pos:
        if pos.count('-')/l < cutoff: # check no. of blanks at pos
            msa_blanks_cutoff.append(pos)
    return list(zip(*msa_blanks_cutoff)) # transpose back and return


def process_alignment(stats_file='stats.csv', min_cov=1.0, min_ident=0.1, max_ident=0.9, max_blank=1.0, db_out='final_alignment.fasta', fmt='fasta'):
    """ Filters and MSA by pairwise identity and coverage. It also filters blank
    columns if over the blank cutoff.

    Parameters:
        stats_file - str: CSV file that contains stats from an MSA processed by get_alignment_stats.py
        min_cov - float: minimum coverage cutoff
        min_ident - float: minimum identity cutoff
        max_ident - float: maximum identity cutoff
        max_blank - float: maximim blank percentage within a position.
        db_out - str: file name of the msa produced.
        fmt - str: format of output file.
    Returns:
        None
    """
    stats = pd.read_csv(stats_file)
    # filter using cutoffs
    stats = stats[(stats['identity'] > min_ident) & ((stats['identity'] < max_ident) | (stats['identity'] == 1)) & (stats['coverage'] > min_cov)]
    seqs = filter_pos_by_blank([list(seq) for seq in stats['seq_aln'].values], cutoff=max_blank)

    # make records list and write output alignment
    recs = [SeqRecord(Seq(''.join(seq)), id=seqid) for seq, seqid in zip(seqs, stats['id'].values)]
    SeqIO.write(recs, open(db_out, 'w'), 'fasta')
    return None
if __name__ == '__main__':
    process_alignment(**vars(args))
