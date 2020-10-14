import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import umap
from matplotlib import pyplot as plt

import sys
import pathlib

source_dir = (pathlib.Path.cwd()/'..')
sys.path.append(str(source_dir))
import utils
from read_config import Config
import examine_model

def tsne_projection(dataset):
    return TSNE().fit_transform(dataset)

def tsne_projection_from_config(config):
    latents = examine_model.get_saved_latent_space_as_numpy(config)
    means = latents[0]
    return tsne_projection(means)

def umap_projection(dataset):
    reducer = umap.UMAP()
    return reducer.fit_transform(dataset)

def umap_projection_from_config(config):
    latents = examine_model.get_saved_latent_space_as_numpy(config)
    means = latents[0]
    
    return umap_projection(means)


def plot_projection(embedding):
    plt.figure()
    x = [row[0] for row in embedding]
    y = [row[1] for row in embedding]
    return sns.scatterplot(x=x, y=y, marker='x', alpha=.4, s=1, color='red')

def plot_latents(config, projection='umap'):
    if projection=='umap':
        points = umap_projection(config)
    elif projection=='tsne':
        points = tsne_projection(config)
    else:
        print("Unrecognized projection")
        return
    return plot_projection(points)