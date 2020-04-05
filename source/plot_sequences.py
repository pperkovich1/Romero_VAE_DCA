import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# local imports
import examine_model
import utils
import vae
from dataloader import MSADataset, OneHotTransform


def save_latent_space_plot(config):
    if config.latent_layer_size != 2:
        raise ValueError(f"Latent space size must be 2 to plot. "
                         f"Got {config.latent_layer_size} instead")
    
    mean, log_vars = examine_model.get_saved_latent_space_as_numpy(
                            config.latent_fullpath)
    
    if (mean.shape[1] != 2):
        raise ValueError(f"Latent Space must have dimension 2. "
                         f"Got dimension: {mean.shape[1]}")
    
    foreground_latent_space = None
    if config.foreground_sequences_filename != "":
        foreground_dataset = MSADataset(config.foreground_sequences_fullpath, 
                                        transform=OneHotTransform(21))
        foreground_latent_space = examine_model.calc_latent_space_from_config(
                dataset=foreground_dataset,
                config=config,
                batch_size= len(foreground_dataset))
        foreground_means, foreground_logvars = \
                examine_model.convert_torch_latent_space_to_numpy(
                    foreground_latent_space)
    
    figsize=(12,12)
    plt.figure(figsize=figsize)
    plt.plot(mean[:, 0], 
             mean[:, 1], 'o', markersize=1, alpha=0.5,
            label=config.model_name)
    plt.title("latent space (projected to two dimensions using VAE)")
    plt.xlabel("z1 (1st dim)")
    plt.ylabel("z2 (2nd dim)")
    if foreground_latent_space is not None:
        plt.plot(foreground_means[:, 0],
                 foreground_means[:, 1], 'ro', markersize=4, 
                 label=config.foreground_sequences_label)
    plt.legend();
    plt.savefig(config.latent_plot_output_fullpath)    

if __name__ == "__main__":
    import argparse
    import read_config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = read_config.Config(args.config_filename)

    print("Saving latent space plot")
    save_latent_space_plot(config)


