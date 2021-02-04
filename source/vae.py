import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

#debug libraries
import time
import resource

#local files
import utils
from dataloader import MSADataset, OneHotTransform


class VAE(torch.nn.Module):
    """ This class originally copied from 
    github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-var.ipynb
    and then modified """
    def __init__(self, input_length, num_hidden, num_latent, activation_func, device):
        """ `input_length`: Number of binary variables 
                            (Length of protein x alphabet size)
            `num_hidden` : size of hidden layers. (If this is a list then 
                            it is expanded to multiple hidden layers on the
                            encoder as well as the decoder)
            `num_latent` : size of latent layer
            `activation_func`: Usually sigmoid
            `device` : whether to run on cpu or gpu
        """
        super(VAE, self).__init__()
        self.input_length = input_length
        
        ### MISC
        if not (isinstance(num_hidden, list)):
            num_hidden = [num_hidden]
        nums = [input_length, *num_hidden, num_latent]
        self.activation_func = activation_func
        self.device = device

        ### ENCODER
        self.hidden_in = [torch.nn.Linear(nums[i],
                                nums[i+1]).to(self.device) for i in
                                range(len(nums)-2)]
        self.z_mean = torch.nn.Linear(nums[-2], num_latent).to(self.device)
        self.z_log_var = torch.nn.Linear(nums[-2], num_latent).to(self.device)
        
        ### DECODER
        self.hidden_out = [torch.nn.Linear(nums[i-1], nums[i-2]).to(self.device)
                                  for i in range(len(nums),2, -1)]
        self.linear_4 = torch.nn.Linear(nums[1], input_length).to(self.device)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def encoder(self, input_images):
        x = input_images
        for layer in self.hidden_in:
            x=self.activation_func(layer(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded
    
    def decoder(self, encoded):
        x = encoded
        for layer in self.hidden_out:
            x = self.activation_func(layer(x))
        x = self.linear_4(x)
        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, input_images):
        z_mean, z_log_var, encoded = self.encoder(input_images)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded

def kl_divergence(z_mean, z_log_var, weight=1):
    '''Computes the Kullback-Leibler divergence. Will return a
    weighted divergence if the parameter is provided.

    Parameters:
    z_mean: tensor - mean tensor from latent space.
    z_log_var: tensor - variance tensor from latent space.
    weight: float - default=1 scales the KLD.
    '''
    kld = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    return weight * kld

def bce(recon_images, input_images, weights):

    return F.binary_cross_entropy(recon_images, input_images, reduction='sum')
                                  #weight=weights, reduction='sum')# weights is incorrect dimension when batch size isn't 1


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
            kld = kl_divergence(z_mean, z_log_var)
            bce_loss = bce(recon_images, input_images, weights)
            loss = kld+bce_loss

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
    dataset = MSADataset(config.aligned_msa_fullpath,
            transform=OneHotTransform(21),
            convert_unknown_aa_to_gap=config.convert_unknown_aa_to_gap)

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


def graph_loss(config):
    from matplotlib import pyplot as plt
    with open(config.loss_fullpath, 'rb') as fh:
        loss = pickle.load(fh)
    plt.plot(loss['loss'])
    plt.title("Training Loss")
    plt.xlabel("Round Number")
    plt.ylabel("Loss")
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
            transform=OneHotTransform(21),
            convert_unknown_aa_to_gap=config.convert_unknown_aa_to_gap)
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
    from read_config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = Config(args.config_filename)

    train_and_save_model(config)

    print ("Saving Graph of loss function")
    # graph_loss(config)

    print ("Saving Latent space")
    # save_latent_space_from_config(config)

