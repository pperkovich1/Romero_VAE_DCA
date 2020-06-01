import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

#debug libraries
import time

#local files
import utils
from train_model import load_model_from_config
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
    # TODO: make sure that the latent space is the same size as the input
    # sequences and in the same order. Batch sampling is not necessarily
    # sequential
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

def calc_reconstruction_identity(model, loader, sample_size, device):
    with torch.no_grad():
        idents = torch.zeros(len(loader.dataset))
        batch_size = loader.batch_size
        left = 0 
        for batch_id, (input_images, weights) in enumerate(loader):
            input_images = input_images.to(device)
            weight = weights.to(device)

            for i in range(sample_size):
                z_mean, z_log_var, encoded, recon_images = model(input_images)
                recon_images = utils.softmax(recon_images)
                ident_matrix = torch.mul(recon_images, input_images)
                recon_idents = ident_matrix.sum(1)
                idents[left:left+batch_size] += recon_idents
            left += batch_size
            if left > 100:
                break
        idents = idents/sample_size
    return idents


def get_reconstruction_identity_from_config(config, batch_size = None,
                                            sample_size = 1):
    # args: sample_size - number of times a vector is passed through model to
    #                            calculate average reconstruction identity
    dataset = MSADataset(config.aligned_msa_fullpath,
            transform=OneHotTransform(21))
    input_length = utils.get_input_length(dataset)
    model = load_model_from_config(input_length=input_length, config=config)
    
    if batch_size is None:
        batch_size = config.batch_size
    loader = DataLoader(dataset = dataset, batch_size = batch_size)

    idents = calc_reconstruction_identity(model, loader, sample_size, 
                                          device=config.device)
    with open(config.reconstruction_identity_fullpath, 'wb') as fh:
        pickle.dump({'idents':idents}, fh)




if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = Config(args.config_filename)

    print ("Saving Graph of loss function")
    # graph_loss(config)

    print ("Saving Latent space")
    # save_latent_space_from_config(config)

    print("Calculating reconstruction identity")
    get_reconstruction_identity_from_config(config)
