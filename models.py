''' Author: Juan R. Diaz Rodriguez
last updated: 2019-05-28 JRD
Some of this code was found on this tutorial:
It was then extended to fit our needs.
'''
# @TODO: Do something about all the copy-pasting here
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

class VAE(nn.Module):
    '''Linear Variational Autoencoder

    Parameters:
    l: int - size of input vector
    hidden_size: int - size of hidden vector. 
    latent_size: int - size of latent(bottleneck) vector
    he1_func: method - specifies which non-linear function is applied
                    to the first hidden layer of the encoder.
    hd1_func: method - specifies which non-linear function is applied
                    to the first decoder hidden layer.
    out_func: method - non-linear function for output.

    Assumes input is already a 1D array. If using images
    they need to be flattened into 1D arrays.
    '''
    def __init__(self, l, hidden_size=400, latent_size=10,
            he1_func=nn.Sigmoid(), hd1_func=nn.Sigmoid(), out_func=nn.Softmax(dim=1), device = None):

        super(VAE, self).__init__()
        self.l = l
        # specify layer structure
        self.fc1 = nn.Linear(l, hidden_size)
        # use for mean
        self.fc2m = nn.Linear(hidden_size, latent_size) 
        self.fc2s = nn.Linear(hidden_size, latent_size)
        # use for standard deviation
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, l)

        #tracking variables
        self.z = None
        self.log_s = None
        self.m = None
        #function methods
        self.he1_func=he1_func
        self.hd1_func=hd1_func
        self.out_func=out_func
        
    def reparameterize(self, log_var, mu):
        '''Applies the reparametrization trick and then samples from
        the gaussian.

        Parameters:
        log_var: tensor - variance tensor from encoder
        mu: tensor - mean tensor from encoder
        '''
        s = torch.exp(0.5*log_var)
        eps = torch.rand_like(s) # generate a standard normal same shape as s
        return eps.mul(s).add_(mu).float() # sample from the gaussian
        
    def forward(self, x_in):
        '''Applies the a full forward pass for the autoencoder.
        Parameters:
        x_in: tensor - one dimensional input tensor.
        '''
        self.encode(x_in)
        self.z = self.reparameterize(self.log_s.float(), self.m.float())
        x_pred = self.decode(self.z.float())
        return x_pred.float(), self.m.float(), self.log_s.float()

    def encode(self, x_in):
        '''Runs the encoder and sets the bottleneck variance and mean
        tensors. Uses the non-linear functions specified in the init
        parameters.

        Parameters:
        x_in: tensor - one dimensional input tensor.
        '''
        x = self.he1_func(self.fc1(x_in.float()))
        # Reparametrization trick.
        self.log_s = self.fc2s(x.float())
        self.m = self.fc2m(x.float())

    def decode(self, z):
        '''Runs the decoder and returns the reconstructed input vector.
        Uses the non-linear functions specified in the init parameters.
        
        Parameters:
        z: tensor - one dimensional sampled tensor. This is the rensor
            sampled from latent space.
        '''
        x = self.hd1_func(self.fc3(z.float()))
        x = self.out_func(self.fc4(x.float()))
        return x

class VAE_double(nn.Module):
    '''Linear Variational Autoencoder

    Parameters:
    l: int - size of input vector
    hidden_size_1: int - size of the first hidden vector 
    hidden_size_2: int - size of the second hidden vector
    latent_size: int - size of latent(bottleneck) vector
    he1_func: method - specifies which non-linear function is applied
                    to the first hidden layer of the encoder.
    he2_func: method - specifies which non-lineare function is applied
                    to the second hidden layer of the encoder
    hd1_func: method - specifies which non-linear function is applied
                    to the first decoder hidden layer.
    hd2_func: method - specifies which non-linear function is applied
                    to the second decoder hidden layer.
    out_func: method - non-linear function for output.

    Assumes input is already a 1D array. If using images
    they need to be flattened into 1D arrays.
    '''
    def __init__(self, l, hidden_size_1=400, hidden_size_2=200, latent_size=10,
            he1_func=nn.Sigmoid(), he2_func=nn.Sigmoid(),
            hd1_func=nn.Sigmoid(), hd2_func=nn.Sigmoid(),  out_func=nn.Softmax(dim=1)):

        super(VAE_double, self).__init__()
        self.l = l
        # specify layer structure
        self.fc1_1 = nn.Linear(l, hidden_size_1)
        self.fc1_2 = nn.Linear(hidden_size_1, hidden_size_2)
        # use for mean
        self.fc2m = nn.Linear(hidden_size_2, latent_size) 
        self.fc2s = nn.Linear(hidden_size_2, latent_size)
        # use for standard deviation
        self.fc3_1 = nn.Linear(latent_size, hidden_size_2)
        self.fc3_2 = nn.Linear(hidden_size_2, hidden_size_1)
        self.fc4 = nn.Linear(hidden_size_1, l)

        #tracking variables
        self.z = None
        self.log_s = None
        self.m = None
        #function methods
        self.he1_func=he1_func
        self.he2_func=he2_func
        self.hd1_func=hd1_func
        self.hd2_func=hd2_func
        self.out_func=out_func
        
    def reparameterize(self, log_var, mu):
        '''Applies the reparametrization trick and then samples from
        the gaussian.

        Parameters:
        log_var: tensor - variance tensor from encoder
        mu: tensor - mean tensor from encoder
        '''
        s = torch.exp(0.5*log_var)
        eps = torch.rand_like(s) # generate a standard normal same shape as s
        return eps.mul(s).add_(mu).float() # sample from the gaussian
        
    def forward(self, x_in):
        '''Applies the a full forward pass for the autoencoder.
        Parameters:
        x_in: tensor - one dimensional input tensor.
        '''
        self.encode(x_in)
        self.z = self.reparameterize(self.log_s.float(), self.m.float())
        x_pred = self.decode(self.z.float())
        return x_pred.float(), self.m.float(), self.log_s.float()

    def encode(self, x_in):
        '''Runs the encoder and sets the bottleneck variance and mean
        tensors. Uses the non-linear functions specified in the init
        parameters.

        Parameters:
        x_in: tensor - one dimensional input tensor.
        '''
        x = self.he1_func(self.fc1_1(x_in.float()))
        x = self.he2_func(self.fc1_2(x));
        # Reparametrization trick.
        self.log_s = self.fc2s(x.float())
        self.m = self.fc2m(x.float())

    def decode(self, z):
        '''Runs the decoder and returns the reconstructed input vector.
        Uses the non-linear functions specified in the init parameters.
        
        Parameters:
        z: tensor - one dimensional sampled tensor. This is the rensor
            sampled from latent space.
        '''
        x = self.hd1_func(self.fc3_1(z.float()))
        x = self.hd2_func(self.fc3_2(x))
        x = self.out_func(self.fc4(x.float()))
        return x

class VAE_flexible(nn.Module):
    '''Linear Variational Autoencoder

    Parameters:
    l: int - size of input vector
    hidden_sizes: [int] - size(s) of hidden vector(s), in order. 
    latent_size: int - size of latent(bottleneck) vector
    he1_func: method - specifies which non-linear function is applied
                    to the hidden layers of the encoder.
    hd1_func: method - specifies which non-linear function is applied
                    to the decoder hidden layers.
    out_func: method - non-linear function for output.

    Assumes input is already a 1D array. If using images
    they need to be flattened into 1D arrays.
    '''
    def __init__(self, l, hidden_sizes=[400], latent_size=10,
            he_funcs=[nn.Sigmoid()], hd_funcs=[nn.Sigmoid()], out_func=nn.Softmax(dim=1)):
        # if hidden_sizes are specified but he_funcs aren't, expands he_funcs to match dim of hidden_sizes
        if len(he_funcs)==1:
            he_funcs = he_funcs*len(hidden_sizes)
        if len(hd_funcs)==1:
            hd_funcs = hd_funcs*len(hidden_sizes)

        super(VAE_flexible, self).__init__()
        self.l = l
        # specify layer structure 
        # # encoder network
        # # # hidden layers
        hidden_sizes.insert(0, l)
        self.fc1 = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
                                  for i in range(len(hidden_sizes)-1)])
        # # # use for mean
        self.fc2m = nn.Linear(hidden_sizes[-1], latent_size) 
        # # # use for standard deviation
        self.fc2s = nn.Linear(hidden_sizes[-1], latent_size)

        # # decoder network
        # # # hidden layers
        hidden_sizes.remove(l)
        hidden_sizes.append(latent_size)
        self.fc3 = nn.ModuleList([nn.Linear(hidden_sizes[i+1], hidden_sizes[i])
                                  for i in range(len(hidden_sizes)-2,-1, -1)])
        # # # final output
        self.fc4 = nn.Linear(hidden_sizes[0], l)

        #tracking variables
        self.z = None
        self.log_s = None
        self.m = None
        #function methods
        self.he_funcs=he_funcs
        self.hd_funcs=hd_funcs
        self.out_func=out_func
        
    def forward(self, x_in):
        '''Applies a full forward pass for the autoencoder.
        Parameters:
        x_in: tensor - one dimensional input tensor.
        '''
        self.encode(x_in)
        self.z = self.reparameterize(self.log_s.float(), self.m.float())
        x_pred = self.decode(self.z.float())
        return x_pred.float(), self.m.float(), self.log_s.float()

    def encode(self, x_in):
        '''Runs the encoder and sets the bottleneck variance and mean
        tensors. Uses the non-linear functions specified in the init
        parameters.

        Parameters:
        x_in: tensor - one dimensional input tensor.
        '''
        x = x_in
        for layer_fc, activation_func in zip(self.fc1, self.he_funcs):
            x = activation_func(layer_fc(x.float()))
        # Reparametrization trick.
        self.log_s = self.fc2s(x.float())
        self.m = self.fc2m(x.float())

    def reparameterize(self, log_var, mu):
        '''Applies the reparametrization trick and then samples from
        the gaussian.

        Parameters:
        log_var: tensor - variance tensor from encoder
        mu: tensor - mean tensor from encoder
        '''
        s = torch.exp(0.5*log_var)
        eps = torch.rand_like(s) # generate a standard normal same shape as s
        return eps.mul(s).add_(mu).float() # sample from the gaussian
        
    def decode(self, z):
        '''Runs the decoder and returns the reconstructed input vector.
        Uses the non-linear functions specified in the init parameters.
        
        Parameters:
        z: tensor - one dimensional sampled tensor. This is the rensor
            sampled from latent space.
        '''
        x = z
        for fc, func in zip(self.fc3, self.hd_funcs):
                x = func(fc(x.float()))
        x = self.out_func(self.fc4(x.float()))
        return x


def vae_loss(input_image, recon_image, mu, log_var):
    '''Calculates binary cross entropy + KLD for a reconstructed tensor.

    Parameters:
    input_image: tensor - input tensor
    recon_image: tensor - reconstructed(estimated) tensor.
    mu: tensor - mean tensor from latent space
    log_var: tensor - variance vector from latent space.
    '''
    BCE = F.binary_cross_entropy(recon_image.float(),
        input_image.float(),
        reduction='sum')
    return kl_divergence(mu, logvar) + BCE.float()

def kl_divergence(mu, log_var, weight=1):
    '''Computes the Kullback-Leibler divergence. Will return a
    weighted divergence if the parameter is provided.

    Parameters:
    mu: tensor - mean tensor from latent space.
    log_var: tensor - variance tensor from latent space.
    weight: float - default=1 scales the KLD.
    '''
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return weight * kld.float() 
