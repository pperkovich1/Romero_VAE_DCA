import torch
from torch.utils.data import DataLoader
import time
import os
import pickle
import numpy as np

#local files
import config
from model import VAE
from dataloader import MSADataset, OneHotTransform

#TODO: use spyder, check memory usage of loss arrays

def train_model(device, model, trainloader, valloader, max_epochs, convergence_limit, learning_rate):
    start_time = time.time()
    min_loss = np.inf
    no_improvement = 0
    train_loss = []
    val_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):
        if not epoch%100:
            print('Epoch: %i\tTime elapsed:%.2f sec'%(epoch, (time.time()-start_time)))

        loss_sum = 0
        if no_improvement > convergence_limit:
            print("convergence at %i iterations" % epoch)
        for batch_id, (input_images, weights) in enumerate(trainloader):
            input_images = input_images.to(device)
            weights = weights.to(device)

            z_mean, z_log_var, encoded, recon_images = model(input_images)
            kld = utils.kl_divergence(z_mean, z_log_var)
            bce = utils.bce(recon_images, input_images, weights)
            loss = kld+bce

            optimizer.zero_grad()
            loss.backward
            optimizer.step()

            loss_sum += torch.sum(loss.detach())
        train_loss.append(loss_sum/len(trainloader))

        #validation
        with torch.no_grad():
            loss_sum = 0
            for batch_id, (input_images, weights) in enumerate(valloader):
                input_images = input_images.to(device)
                weights = weights.to(device)

                z_mean, z_log_var, encoded, recon_images = model(input_images)
                kld = utils.kl_divergence(z_mean, z_log_var)
                bce = utils.bce(recon_images, input_images, weights)
                loss_sum += torch.sum(kld+bce)
            loss=loss_sum/len(valloader)
            if min_loss < loss:
                no_improvement += 1
                print('no improvement')
            else:
                min_loss = loss
                no_improvement = 0
                val_loss.append(loss)
                print('improvement')

    torch.save(model.state_dict(), "model.pt")
    pickle.dump({'validation loss': val_loss, 'train_loss':train_loss}, open('loss.pkl', 'wb'))

def main():

    input_length = config.input_length
    num_hidden = config.num_hidden
    num_latent = config.num_latent
    activation_func = config.activation_func
    learning_rate = config.learning_rate
    device = config.device
    model = VAE(input_length, num_hidden, num_latent, activation_func, device)


    if os.path.exists(config.prev_model):
        print("Loading saved model...")
        model.load_state_dict(torch.load(config.prev_model))

    model.to(device)

    dataset = MSADataset(config.msa, transform=OneHotTransform(21))
    trainsize = int(len(dataset)*.9)
    valsize = len(dataset) - trainsize
    (trainset, valset) = torch.utils.data.random_split(dataset, (trainsize, valsize))


    batch_size = config.batch_size
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size)
    valloader = DataLoader(dataset=valset, batch_size=batch_size)



    max_epochs = config.max_epochs
    convergence_limit = config.convergence_limit
    train_model(device, model, trainloader, valloader, max_epochs, convergence_limit, learning_rate)

if __name__=='__main__':
    main()
