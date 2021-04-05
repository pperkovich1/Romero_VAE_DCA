import pickle
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

#local files
from dataloader import MSADataset, OneHotTransform
import vae
import utils

def calc_latent_space(model, loader, device):
    latent_vecs = []

    with torch.no_grad():
        for batch_id, (input_images, weights) in enumerate(loader):
            input_images = input_images.to(device)

            z_mean, z_log_var, encoded, recon_images = model(input_images)
            #TODO: break up batches
            for m, v, in zip(z_mean, z_log_var):
                latent_vecs.append((m, v))
    return latent_vecs

def calc_latent_space_from_config(dataset, config, batch_size = None):
    """Takes an MSADataset and config file and returns the latent space """
    input_length = utils.get_input_length(dataset)
    model = vae.load_model_from_config(input_length=input_length, config=config,
                reload_saved_model=True)

    if batch_size is None:
        batch_size = config.batch_size
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
            sampler=None)

    return calc_latent_space(model, loader, device=config.device)

def get_saved_latent_space_as_numpy(latent_fullpath):
    ret = pd.read_pickle(latent_fullpath)
    return ret["means"], ret["log_vars"]

def get_latent_space_from_config(config, input_filename=None,
        output_filename=None):
    if input_filename is None:
        input_filename = config.aligned_msa_fullpath
    logging.info(f"Reading sequence data from {input_filename}")
    dataset = MSADataset(input_filename, 
            transform=OneHotTransform(21),
            convert_unknown_aa_to_gap=config.convert_unknown_aa_to_gap)
    latent_vecs = calc_latent_space_from_config(dataset, config)

    seqs = [str(s) for s in dataset.raw_data]
    means = torch.stack(tuple(v[0] for v in latent_vecs)).numpy()
    log_vars = torch.stack(tuple(v[1] for v in latent_vecs)).numpy()

    latent_data = {"seqs":seqs, "means":means, "log_vars":log_vars}
    return latent_data

if __name__=='__main__':
    import time
    import argparse
    from read_config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    parser.add_argument("-i", "--input_filename",
                    help="Input fasta/aln file with sequences",
                    default=None,
                    required=False) # default is false
    parser.add_argument("-o", "--output_filename",
                    help="output latent space",
                    default=None,
                    required=False) # default is false
    args = parser.parse_args()

    config = Config(args.config_filename)
    logging.basicConfig(level=getattr(logging, config.log_level))

    logging.info(f"Calculating latent space")
    latent_data = get_latent_space_from_config(config,
            input_filename=args.input_filename)
    output_filename = args.output_filename
    if output_filename is None:
        output_filename = config.latent_fullpath
    logging.info(f"Writing latent space to {output_filename}")
    # latent_data is not a pandas Dataframe but Pandas is convient for pickling
    # anything. The output file will be compressed or not depending on
    # extension.
    pd.to_pickle(latent_data, output_filename)


