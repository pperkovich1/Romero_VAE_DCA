import pickle
import logging

import torch

import matplotlib.pyplot as plt

# local imports
import vae_latent_space


def save_latent_space_plot(config):
    if config.latent_layer_size != 2:
        raise ValueError(f"Latent space size must be 2 to plot. "
                         f"Got {config.latent_layer_size} instead")
    
    mean, log_vars = vae_latent_space.get_saved_latent_space_as_numpy(
                            config.latent_fullpath)
    
    if (mean.shape[1] != 2):
        raise ValueError(f"Latent Space must have dimension 2. "
                         f"Got dimension: {mean.shape[1]}")
    
    foreground_latent_space = None
    foreground_means = None
    if config.foreground_sequences_filename:
        foreground_latent_space = \
            vae_latent_space.get_latent_space_from_config(config,
                input_filename=config.foreground_sequences_filename)
        foreground_means = foreground_latent_space["means"]
        foreground_logvars = foreground_latent_space["log_vars"]
    
    logging.info("Plotting latent space")
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

    # Now save all the elements of the plot to an archive
    plotd = {'label': config.model_name,
             'background_latent': mean,
             'foreground_latent': foreground_means}
    with config.latent_plot_archive_fullpath.open('wb') as fh:
        pickle.dump(plotd, fh)

if __name__ == "__main__":
    import argparse
    import read_config

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = read_config.Config(args.config_filename)
    logging.basicConfig(level=getattr(logging, config.log_level))

    print("Saving latent space plot")
    save_latent_space_plot(config)


