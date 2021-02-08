import os
import pickle
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

#debug libraries
import time

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
        ### FIXME: TODO make this a sequential list or module list
        self.hidden_in = torch.nn.ModuleList([torch.nn.Linear(nums[i],
                                nums[i+1]).to(self.device) for i in
                                range(len(nums)-2)]).to(self.device)
        self.z_mean = torch.nn.Linear(nums[-2], num_latent).to(self.device)
        self.z_log_var = torch.nn.Linear(nums[-2], num_latent).to(self.device)
        
        ### DECODER
        self.hidden_out = torch.nn.ModuleList([torch.nn.Linear(nums[i-1],
                nums[i-2]).to(self.device) for i in 
                range(len(nums),2, -1)]).to(self.device)
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


def train_vae_model(model, loader, epochs, learning_rate, device,
        convergence_limit=99999):
    """ Convergence limit - place holder in case we want to train based on loss
        improvement
    """ 
    start_time = time.time()
    min_loss = 999999
    no_improvement = 0
    loss_history = np.zeros(epochs, dtype=np.float)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logging.info('Max memory usage:%s'%(utils.get_max_memory_usage()))
    len_loader = len(loader)
    for epoch in range(epochs):

        if no_improvement > convergence_limit:
            print("convergence at %i iterations" % epoch)

        for batch_id, (input_images, weights) in enumerate(loader):
            input_images = input_images.to(device)
            weights = weights.to(device)

            z_mean, z_log_var, encoded, recon_images = model(input_images)
            kld = kl_divergence(z_mean, z_log_var)
            bce_loss = bce(recon_images, input_images, weights)
            loss = kld + bce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history[epoch] += loss.item()
        loss_history[epoch] /= len_loader
        if loss_history[epoch] < min_loss:
            min_loss = loss_history[epoch]
            no_improvement = 0
        else:
            no_improvement +=1 
        if not epoch%1:
            logging.info(
                "Epoch: %i Loss: %.2f Time elapsed:%7.2f min, memory=%s" % (
                    epoch, loss_history[epoch], (time.time()-start_time)/60,
                    utils.get_max_memory_usage()))
    ret = {"loss_history":loss_history, "optimizer":optimizer}
    return ret


def load_model_from_path(model_fullpath, input_length, hidden_layer_size,
        latent_layer_size, activation_func, device):
    model = VAE(input_length, hidden_layer_size, latent_layer_size, 
            activation_func, device)
    if model_fullpath and os.path.exists(model_fullpath):
        logging.info(f"Loading saved model from {model_fullpath}")
        model.load_state_dict(torch.load(model_fullpath))
        # TODO: Do we need to run model.eval() here? see,
        # https://pytorch.org/tutorials/beginner/saving_loading_models.htm
    model.to(device)
    return model

def load_model_from_config(input_length, config, reload_saved_model=False):
    model_fullpath = ""
    if reload_saved_model:
        model_fullpath = config.model_fullpath
    model = load_model_from_path(model_fullpath = model_fullpath,
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
        logging.info("Found weights in sampler! Enabling weighted sampling.")
        sampler = WeightedRandomSampler(weights=weights,
                num_samples=num_samples)
    else:
        logging.info("Can't find weights. No weighted sampling will be done.")
    return sampler

def train_vae_model_from_config(config, reload_saved_model=False):
    dataset = MSADataset(config.aligned_msa_fullpath,
            transform=OneHotTransform(21),
            convert_unknown_aa_to_gap=config.convert_unknown_aa_to_gap)

    input_length = utils.get_input_length(dataset)
    model = load_model_from_config(input_length=input_length, config=config)
    batch_size = config.batch_size
    sampler = load_sampler(len(dataset), config)
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
            sampler=sampler)

    learning_rate = config.learning_rate
    epochs = config.epochs
    ret = train_vae_model(model=model, loader=loader, epochs=epochs, 
            learning_rate=learning_rate, device=config.device)

    ret["model"]=model # add model to return value
    return ret

if __name__=='__main__':
    import argparse
    from read_config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    parser.add_argument("-r", "--reload_saved_model",
                    help="Start training from previously saved model",
                    action='store_true') # default is false
    args = parser.parse_args()


    config = Config(args.config_filename)
    logging.basicConfig(level=getattr(logging, config.log_level))

    reload_saved_model = args.reload_saved_model # default is false

    ret = train_vae_model_from_config(config,
            reload_saved_model=reload_saved_model)
    torch.save(ret["model"].state_dict(), config.model_fullpath)
    with open(config.loss_fullpath, 'wb') as fh:
        pickle.dump({'loss':ret["loss_history"]}, fh)

    print ("Saving Graph of loss function")
    utils.plot_loss_curve(losses=ret["loss_history"],
            annotatation_str=str(ret["optimizer"]),
            save_fig_path = config.lossgraph_fullpath,
            model_name = config.model_name)

    #print ("Saving Latent space")
    # save_latent_space_from_config(config)

