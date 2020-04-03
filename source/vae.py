import torch
from torch.utils.data import DataLoader
import os
import pickle

#debug libraries
import time
import resource

#local files
import utils
from model import VAE
from dataloader import MSADataset, OneHotTransform
from read_config import Config

def train_model(device, model, loader, max_epochs, learning_rate, model_filename, convergence_limit=99999):
    ''' Convergence limit - place holder in case we want to train based on loss improvement '''
    if os.path.exists(model_filename):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_filename))

    start_time = time.time()
    min_loss = 999999
    no_improvement = 0
    loss_history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):
        print('Max  memory usage:%d'%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if not epoch%1:
            print('Epoch: %i\tTime elapsed:%.2f sec'%(epoch, (time.time()-start_time)))
        loss_history.append(0)

        if no_improvement > convergence_limit:
            print("convergence at %i iterations" % epoch)

        for batch_id, (input_images, weights) in enumerate(loader):
            input_images = input_images.to(device)
            weights = weights.to(device)

            z_mean, z_log_var, encoded, recon_images = model(input_images)
            kld = utils.kl_divergence(z_mean, z_log_var)
            bce = utils.bce(recon_images, input_images, weights)
            loss = kld+bce

            optimizer.zero_grad()
            loss.backward
            optimizer.step()

            loss=loss.detach()
            loss_history[epoch] +=loss
        loss_history[epoch] = loss_history[epoch]/len(loader)
        if loss_history[epoch] < min_loss:
            min_loss = loss_history[epoch]
            no_improvement = 0
        else:
            no_improvement +=1 

    torch.save(model.state_dict(), model_filename)
    pickle.dump({'loss':loss_history}, open('loss.pkl', 'wb'))

def main():
    config = Config('../config.yaml')

    dataset = MSADataset(config.aligned_msa_fullpath, transform=OneHotTransform(21))

    input_length = utils.get_input_length(dataset)
    hidden = config.hidden_layer_size
    latent = config.latent_layer_size
    activation_func = config.activation_function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(input_length, hidden, latent, activation_func, device)

    model.to(device)

    batch_size = config.batch_size
    loader = DataLoader(dataset=dataset, batch_size=batch_size)



    learning_rate = config.learning_rate
    epochs = config.epochs
    train_model(device, model, loader, epochs, learning_rate, config.model_filename)

if __name__=='__main__':
    main()
