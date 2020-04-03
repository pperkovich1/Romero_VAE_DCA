import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import pickle

#debug libraries
import time

#local files
import utils
from model import VAE
from dataloader import MSADataset, OneHotTransform
from read_config import Config

def graphLoss(config):
    loss = pickle.load(open('loss.pkl', 'rb'))
    plt.plot(loss['loss'])
    plt.savefig('loss.png', bbox_inches='tight')


def sampleLatentSpace(model, loader, device):
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
    pickle.dump({'latent':latent_vecs}, open('latent.pkl', 'wb'))

def getLatentSpace(config):
    dataset = MSADataset(config.aligned_msa_fullpath, transform=OneHotTransform(21))

    input_length = utils.get_input_length(dataset)
    hidden = config.hidden_layer_size
    latent = config.latent_layer_size
    activation_func = config.activation_function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(input_length, hidden, latent, activation_func, device)

    model.load_state_dict(torch.load(config.model_name))
    model.to(device)
    
    batch_size = config.batch_size
    loader = DataLoader(dataset=dataset, batch_size=batch_size)


    sampleLatentSpace(model, loader, device)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = Config(args.config_filename)

    getLatentSpace(config)
    graphLoss(config)