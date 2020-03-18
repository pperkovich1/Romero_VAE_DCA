import torch
from torch.utils.data import DataLoader
import time
import utils
import os
import pickle

#debug libraries
import resource

#local files
import config
from model import VAE
from dataloader import MSADataset, OneHotTransform

def train_model(device, model, trainloader, valloader, max_epochs, convergence_limit, learning_rate):
        start_time = time.time()
        min_loss = 999999
        no_improvement = 0
        train_loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(max_epochs):
                print('Max  memory usage:%d'%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                print(len(train_loss))
                if not epoch%10:
                        print('Epoch: %i\tTime elapsed:%.2f sec'%(epoch, (time.time()-start_time)))
                train_loss.append(0)

                if no_improvement > convergence_limit:
                        print("convergence at %i iterations" % epoch)

                batch_count = 0
                for batch_id, (input_images, weights) in enumerate(trainloader):
                        batch_count=batch_count+1
                        if not batch_count%500:
                                print("batch count:%d"%batch_count)
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
                        train_loss[epoch] +=loss
                train_loss[epoch] = train_loss[epoch]/len(trainloader)
                if train_loss[epoch] < min_loss:
                    min_loss = train_loss[epoch]
                    no_improvement = 0
                else:
                    no_improvement +=1 

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
