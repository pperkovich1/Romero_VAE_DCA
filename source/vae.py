import torch
from torch.utils.data import DataLoader
import os
import pickle

#debug libraries
import time
import resource

#local files
import config
import utils
from model import VAE
from dataloader import MSADataset, OneHotTransform

def train_model(device, model, loader, max_epochs, convergence_limit, learning_rate):
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

        torch.save(model.state_dict(), "model.pt")
        pickle.dump({'loss':loss_history}, open('loss.pkl', 'wb'))

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

        batch_size = config.batch_size
        loader = DataLoader(dataset=dataset, batch_size=batch_size)



        max_epochs = config.max_epochs
        convergence_limit = config.convergence_limit
        train_model(device, model, loader, max_epochs, convergence_limit, learning_rate)

if __name__=='__main__':
        main()
