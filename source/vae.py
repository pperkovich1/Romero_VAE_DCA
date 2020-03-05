import torch
from torch.utils.data import DataLoader
import time
import utils
import os
import pickle

#local files
import config
from model import VAE
from dataloader import MSADataset, OneHotTransform

#TODO: use spyder, check memory usage of loss arrays

def train_model(device, model, trainloader, valloader, max_epochs, convergence_limit, learning_rate):
    start_time = time.time()
    min_loss = 999999
    no_improvement = 0
    train_loss = []
    val_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):

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
