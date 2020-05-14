import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import pickle

#debug libraries
import time
import resource

#local files
import utils
from model import VAE
from dataloader import MSADataset, OneHotTransform


def train_model(device, model, loader, max_epochs, learning_rate,
        model_fullpath, loss_fullpath, convergence_limit=99999):
    """ Convergence limit - place holder in case we want to train based on loss
        improvement
    """ 
    start_time = time.time()
    min_loss = 999999
    no_improvement = 0
    loss_history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):
        print('Max  memory usage:%d'%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if not epoch%1:
            print('Epoch: %i\tTime elapsed:%.2f sec'%(epoch, (time.time()-start_time)))
        loss_history.append(0)

        if no_improvement > convergence_limit:
            print("convergence at %i iterations" % epoch)

        for batch_id, (input_images, weights) in enumerate(loader):
            input_images = input_images.to(device)
            weights = weights.to(device)

            z_mean, z_log_var, encoded, recon_images = model(input_images)
            kld = utils.kl_divergence(z_mean, z_log_var)
            bce = utils.bce(recon_images, input_images, weights)
            loss = kld+bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss=loss.detach()
            loss_history[epoch] +=loss
        loss_history[epoch] = loss_history[epoch]/len(loader)
        if loss_history[epoch] < min_loss:
            min_loss = loss_history[epoch]
            no_improvement = 0
        else:
            no_improvement +=1 

    torch.save(model.state_dict(), model_fullpath)
    with open(loss_fullpath, 'wb') as fh:
        pickle.dump({'loss':loss_history}, fh)


def load_model_from_path(model_fullpath, input_length, hidden_layer_size,
        latent_layer_size, activation_func, device):
    model = VAE(input_length, hidden_layer_size, latent_layer_size, 
            activation_func, device)
    if os.path.exists(model_fullpath):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_fullpath))
        # TODO: Do we need to run model.eval() here? see,
        # https://pytorch.org/tutorials/beginner/saving_loading_models.htm
    model.to(device)
    return model

def load_model_from_config(input_length, config):
    model = load_model_from_path(model_fullpath = config.model_fullpath,
            input_length = input_length,
            hidden_layer_size = config.hidden_layer_size,
            latent_layer_size = config.latent_layer_size,
            activation_func = config.activation_function,
            device = config.device)
    return model

def load_sampler(num_samples, config):
    sampler = None
    if config.weights_fullpath.is_file():
        weights = np.load(config.weights_fullpath)
        sampler = WeightedRandomSampler(weights=weights,
                                  num_samples=num_samples)
    else:
        print("Weights do not exist. No weighted sampling will be done.")
    return sampler

def train_and_save_model(config):
    dataset = MSADataset(config.aligned_msa_fullpath, transform=OneHotTransform(21))

    input_length = utils.get_input_length(dataset)
    model = load_model_from_config(input_length=input_length, config=config)
    batch_size = config.batch_size
    sampler = load_sampler(len(dataset), config)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

    learning_rate = config.learning_rate
    epochs = config.epochs
    train_model(config.device, model, loader, epochs, learning_rate, 
            config.model_fullpath, config.loss_fullpath)
    return None



if __name__=='__main__':
    import argparse
    from read_config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = Config(args.config_filename)

    train_and_save_model(config)
