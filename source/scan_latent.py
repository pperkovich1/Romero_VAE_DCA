# libraries
import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

# local files
import utils
from read_config import Config
import examine_model

config = Config('../config_1000_15.yaml')
dataset = utils.get_dataset_from_config(config)
model = utils.load_model_from_config(config)

latent = pickle.load(open(config.latent_fullpath, 'rb'))['latent']
means = latent[0]
log_vars = latent[1]

seq_index = 1
probe_means = means[seq_index]
probe_log_vars = log_vars[seq_index]

print(probe_log_vars)
print(torch.mean(probe_log_vars))



all_recons = []

sample_size = 100
with torch.no_grad():
    free_outputs = []

    # free variation
    for i in range(sample_size):
        encoding = model.reparameterize(probe_means, probe_log_vars)
        free_output = model.decoder(encoding)
        free_outputs.append(free_output)
    free_outputs = torch.stack(free_outputs) 
    free_recons = utils.softmax(free_outputs)
    all_recons.append(free_recons)

    # no variation
    fixed_log_vars = torch.full(probe_log_vars.size(), -float('inf'))
    encoding = model.reparameterize(probe_means, fixed_log_vars)
    fixed_outputs = [model.decoder(encoding)] # still turn into 2d array for consistency
    fixed_outputs = torch.stack(fixed_outputs)
    fixed_recons = utils.softmax(fixed_outputs)
    all_recons.append(fixed_recons)

    for free_var in range(len(probe_log_vars)):
        fixed_log_vars = torch.full(probe_log_vars.size(), -float('inf'))
        fixed_log_vars[free_var] = probe_log_vars[free_var]

        outputs = []
        for i in range(sample_size):
            encoding = model.reparameterize(probe_means, fixed_log_vars)
            output = model.decoder(encoding)
            outputs.append(output)
        # converst list of tensors to a 2D tenor
        outputs = torch.stack(outputs) 
        recons = utils.softmax(outputs)
        all_recons.append(recons)

base = torch.transpose(all_recons[1][0].reshape(-1, 21), 0, 1)
for i, recons in enumerate(all_recons):
    plt.subplot(len(all_recons), 1, i+1)
    freqs = examine_model.seqs_to_freqs(recons)
    print(freqs)
    print(base)
    if i>1:
        freqs = freqs - base
    plt.ylabel(i)


plt.savefig('full_freqs.png', dpi=500) 
