import torch.nn.functional as F
import torch

def get_input_length(dataset):
    sample = dataset.__getitem__(0)[0]
    return len(sample)

def vae_loss(input_image, recon_image, z_mean, z_log_var):
    '''Calculates binary cross entropy + KLD for a reconstructed tensor.

    Parameters:
    input_image: tensor - input tensor
    recon_image: tensor - reconstructed(estimated) tensor.
    z_mean: tensor - mean tensor from latent space
    z_log_var: tensor - variance vector from latent space.
    '''
    BCE = F.binary_cross_entropy(recon_image.float(),
        input_image.float(),
        reduction='sum')
    return kl_divergence(z_mean, z_log_var) + BCE.float()

def kl_divergence(z_mean, z_log_var, weight=1):
    '''Computes the Kullback-Leibler divergence. Will return a
    weighted divergence if the parameter is provided.

    Parameters:
    z_mean: tensor - mean tensor from latent space.
    z_log_var: tensor - variance tensor from latent space.
    weight: float - default=1 scales the KLD.
    '''
    kld = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    return weight * kld.float()

def bce(recon_images, input_images, weights):

    return F.binary_cross_entropy(recon_images, input_images, reduction='sum')
                                  #weight=weights, reduction='sum')# weights is incorrect dimension when batch size isn't 1

def softmax(recon_images):
    #each row of recon_images is a sequence
    for seq in recon_images:
        for i in range(len(seq)//21):
            left = i*21
            right = (i+1)*21
            x, index = seq[left:right].max(0)
            seq[left:right]= 0
            seq[left+index] = 1
    return recon_images

import pickle
import torch
import io

class cpkl(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_to_cpu(path):
    file = open(path, 'rb')
    return cpkl.cpkl(file).load()

from  dataloader import MSADataset, OneHotTransform
def get_dataset_from_config(config):
    return MSADataset(config.aligned_msa_fullpath, transform=OneHotTransform(21))

from model import VAE
import os
def load_model_from_path(model_fullpath, input_length, hidden_layer_size,
        latent_layer_size, activation_func, device):
    model = VAE(input_length, hidden_layer_size, latent_layer_size, 
            activation_func, device)
    if os.path.exists(model_fullpath):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_fullpath, map_location=device))
        # TODO: Do we need to run model.eval() here? see,
        # https://pytorch.org/tutorials/beginner/saving_loading_models.htm
    model.to(device)
    return model

def load_model_from_config(config):
    dataset = get_dataset_from_config(config)
    input_length = get_input_length(dataset)

    model = VAE(input_length, config.hidden_layer_size, config.latent_layer_size,
                config.activation_function, config.device)
    if os.path.exists(config.model_fullpath):
        print("Loading saved model...")
        model.load_state_dict(torch.load(config.model_fullpath, map_location=config.device))
        # TODO: Do we need to run model.eval() here? see,
        # https://pytorch.org/tutorials/beginner/saving_loading_models.htm
    # model.to(config.device)

    return model

