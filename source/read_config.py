import errno
import os

import yaml
import pathlib


dataset_dir = "../sequence_sets"

def load_yaml(yaml_filename):
    with open(yaml_filename, 'r') as fh:
        yaml_data = yaml.safe_load(fh)
    return yaml_data

def print_output_for_chtc(yaml_data):

    input_filename = yaml_data['aligned_msa_filename']
    threshold = yaml_data['reweighting_threshold']

    input_path = dataset_dir / path.Path(input_filename)

    if not input_path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                input_path)

    if input_path.suffix != ".fasta":
        raise ValueError("aligned_msa_filename should end in .fasta.\n"
                         "current value : ", input_filename)

    print(input_path.stem, ",", str(threshold))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputconfig", 
                    help="input config file in yaml format")
    parser.add_argument("-c", "--printchtc", 
                    help="output print output for chtc")
    parser.add_argument("-d", "--datasetdir", 
                    help="directory where dataset is stored")

    args = parser.parse_args()
    if args.datasetdir is not None:
        global dataset_dir
        dataset_dir = args.datasetdir

    yaml_data = load_yaml(parser.inputconfig)
    print_output_for_chtc(yaml_data)
