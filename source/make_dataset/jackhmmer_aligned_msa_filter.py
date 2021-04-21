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
    parser.add_argument("-q", "--query_fasta_filename",
                    help="query sequence (WT) in fasta format",
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

    logging.info(f"Reading query fasta file")
    query_fasta = Bio.AlignIO.read(args.query_fasta_filename, format="fasta")
    query = query_fasta[0]
    logging.info(str(query))

    query_seq_msa = None # pointer to query sequence in the MSA
    if query.seq == msa[-1].seq.ungap('-'):
        logging.info(f"Found query sequence at last record of MSA")
        query_seq_msa = msa[-1]
    elif query.seq == msa[0].seq.ungap('-'):
        logging.info(f"Found query sequence at first record of MSA")
        query_seq_msa = msa[0]

    if query_seq_msa is None:
        raise SystemExit("Cannot find query sequence as first or last record in MSA")
    logging.info(f"Query sequence in MSA: {query_seq_msa.name}")
    logging.info(f"Query sequence length in MSA: {len(query_seq_msa)}")
    logging.info(str(query_seq_msa))

    # Find amino acid positions in query sequence that are not gaps
    match_idx = [i for i,a in enumerate( query_seq_msa.seq ) if a != "-"]
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

