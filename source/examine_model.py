import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

#debug libraries
import time

#local files
import utils
from vae import load_model_from_config
from dataloader import MSADataset, OneHotTransform
from read_config import Config

def graph_loss(config):
    with open(config.loss_fullpath, 'rb') as fh:
        loss = pickle.load(fh)
    plt.plot(loss['loss'])
    plt.savefig(config.lossgraph_fullpath, bbox_inches='tight')


def calc_latent_space(model, loader, device):
    start_time = time.time()
    latent_vecs = []

    with torch.no_grad():
        for batch_id, (input_images, weights) in enumerate(loader):
            input_images = input_images.to(device)
            weights = weights.to(device)

            z_mean, z_log_var, encoded, recon_images = model(input_images)
            #TODO: break up batches
            for m, v, in zip(z_mean, z_log_var):
                latent_vecs.append((m, v))
    return latent_vecs

def calc_latent_space_from_config(dataset, config, 
        batch_size = None):
    """Takes an MSADataset and config file and returns the latent space """
    input_length = utils.get_input_length(dataset)
    model = load_model_from_config(input_length=input_length, config=config)

    if batch_size is None:
        batch_size = config.batch_size
    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    return calc_latent_space(model, loader, device=config.device)

def save_latent_space_from_config(config):
    # TODO: save the latent space as a numpy array
    dataset = MSADataset(config.aligned_msa_fullpath, 
            transform=OneHotTransform(21))
    latent_vecs = calc_latent_space_from_config(dataset, config)
    with open(config.latent_fullpath, 'wb') as fh:
        pickle.dump({'latent':latent_vecs}, fh)

def convert_torch_latent_space_to_numpy(latent_vecs):
    means = torch.stack(tuple(v[0] for v in latent_vecs)).numpy()
    log_vars = torch.stack(tuple(v[1] for v in latent_vecs)).numpy()
    return means, log_vars

def get_saved_latent_space_as_numpy(latent_fullpath):
    with open(latent_fullpath, 'rb') as fh:
        vecs = pickle.load(fh)
    return convert_torch_latent_space_to_numpy(vecs['latent'])

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = Config(args.config_filename)

    print ("Saving Graph of loss function")
    graph_loss(config)

    print ("Saving Latent space")
    save_latent_space_from_config(config)
