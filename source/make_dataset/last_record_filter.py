""" Filter aligned fasta (MSA) file based on last record (query sequence) """

import functools
import operator
import logging
import pathlib

import Bio
import Bio.AlignIO
import Bio.Data
import Bio.SeqIO


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_msa_filename",
                    help="input msa in aligned fasta format",
                    required=True)
    parser.add_argument("-o", "--output_msa_filename",
                    help="output msa in aligned fasta format",
                    required=True)
    parser.add_argument("-f", "--fraction_cutoff",
                    help="Drop sequences with length less than this fraction"
                         " of query sequence",
                    type=float,
                    default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Reading alignment file")
    msa = Bio.AlignIO.read(args.input_msa_filename, format="fasta")
    logging.info(f"Number of sequences in alignment file: {len(msa)}")

    fraction_cutoff =  args.fraction_cutoff

    last_seq = msa[-1]
    # last line should be the fasta sequence we did the query with
    logging.info(f"Query sequence: {last_seq.name}")
    logging.info(f"Query sequence length: {len(last_seq)}")

    # Find amino acid positions in query sequence that are not gaps
    match_idx = [i for i,a in enumerate( last_seq.seq ) if a != "-"]
    prot_length = len(match_idx)
    logging.info(f"Number of columns saved: {prot_length}")

    ## Smaller msa with only match columns
    logging.info("Filtering columns to match query sequence") 
    small_msa = functools.reduce(operator.add, (msa[:, i:(i+1)] 
                        for i in match_idx))
    logging.info(f"Number of sequences left : {len(small_msa)}")

    logging.info("Filtering sequences on length") 
    # filter resulting proteins on their length
    length_filter_msa = Bio.Align.MultipleSeqAlignment([
        r for r in small_msa if 
        len(r.seq.ungap("-")) >= fraction_cutoff*prot_length ])
    logging.info(f"Number of sequences left : {len(length_filter_msa)}")

    logging.info("Filtering out sequences invalid amino acids") 
    invalid_map = str.maketrans('', '', Bio.Data.IUPACData.protein_letters + '-')
    clean_msa = Bio.Align.MultipleSeqAlignment([
        r for r in length_filter_msa if 
        not len(str(r.seq).translate(invalid_map))]) 
    logging.info(f"Number of sequences left : {len(clean_msa)}")

    Bio.SeqIO.write(clean_msa, args.output_msa_filename, "fasta")

