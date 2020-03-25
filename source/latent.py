import torch
from torch.utils.data import DataLoader
import pickle

#debug libraries
import time

#local files
import config
import utils
from model import VAE
from dataloader import MSADataset, OneHotTransform


def getLatentSpace(model, loader, device):
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

def main():
    input_length = config.input_length
    num_hidden = config.num_hidden
    num_latent = config.num_latent
    activation_func = config.activation_func
    learning_rate = config.learning_rate
    device = config.device
    model = VAE(input_length, num_hidden, num_latent, activation_func, device)

    model.load_state_dict(torch.load(config.prev_model))
    model.to(device)
    
    batch_size = config.batch_size
    dataset = MSADataset(config.msa, transform=OneHotTransform(21), size_limit = None)
    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    getLatentSpace(model, loader, device)

if __name__=='__main__':
    main()