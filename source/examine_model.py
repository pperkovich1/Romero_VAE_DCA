import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

#debug libraries
import time

#local files
import utils
from model import VAE
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

def save_latent_space(config):
    dataset = MSADataset(config.aligned_msa_fullpath, transform=OneHotTransform(21))

    input_length = utils.get_input_length(dataset)
    hidden = config.hidden_layer_size
    latent = config.latent_layer_size
    activation_func = config.activation_function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(input_length, hidden, latent, activation_func, device)

    model.load_state_dict(torch.load(config.model_fullpath))
    model.to(device)
    
    batch_size = config.batch_size
    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    latent_vecs = calc_latent_space(model, loader, device)
    with open(config.latent_fullpath, 'wb') as fh:
        pickle.dump({'latent':latent_vecs}, fh)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = Config(args.config_filename)

    save_latent_space(config)
    graph_loss(config)
