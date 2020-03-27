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