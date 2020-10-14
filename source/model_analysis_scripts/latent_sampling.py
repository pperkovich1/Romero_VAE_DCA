import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl
import torch

import sys
import pathlib

source_dir = (pathlib.Path.cwd()/'..')
sys.path.append(str(source_dir))
import utils
from read_config import Config
import examine_model


# generates a library by sampling from standard Gaussian by default
# returns a library of one-hot encoded sequences
def generate_library(config, latent_index = None, library_size = 1000):
    model = utils.load_model_from_config(config)
    dataset = utils.get_dataset_from_config(config)
    
    if not latent_index is None:
        latents = pkl.load(open(config.latent_fullpath, 'rb'))['latent']
        latent_distribution = latents[latent_index]
    else:
        latent_distribution = torch.stack([torch.zeros(config.latent_layer_size), torch.ones(config.latent_layer_size)])
    
    means = latent_distribution[0]
    log_vars = latent_distribution[1]
    
    with torch.no_grad():
        outputs = []
        for i in range(library_size):
            encoding = model.reparameterize(means, log_vars)
            output = model.decoder(encoding)
            outputs.append(output)
        # there will probably by issues because previous code has outputs as a pytorch tensor
        outputs = torch.stack(outputs)
        library = utils.softmax(outputs).numpy()
    return library 