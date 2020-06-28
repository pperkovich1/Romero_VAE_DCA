# libraries
import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

# local files
import utils
from train_model import load_model_from_config
from read_config import Config
import examine_model

config = Config('../config_1000_15.yaml')
dataset = utils.get_dataset_from_config(config)
model = utils.load_model_from_config(config)

latent = pickle.load(open(config.latent_fullpath, 'rb'))['latent']
means = latent[0]
log_vars = latent[1]

probe_means = means[0]
probe_log_vars = log_vars[0]

print(probe_log_vars)
print(torch.mean(probe_log_vars))


sample_size = 15
with torch.no_grad():
    outputs = []
    for i in range(sample_size):
        encoding = model.reparameterize(probe_means, probe_log_vars)
        output = model.decoder(encoding)
        outputs.append(output)
    outputs = torch.stack(outputs)

    recons = utils.softmax(outputs)
    freqs = examine_model.plot_seq_freq_heatmap(recons)

    for free_var in range(len(probe_log_vars)):
        plt.subplot(15, 1, free_var+1)
        print(free_var)
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

        freqs = examine_model.plot_seq_freq_heatmap(recons, image_name="freq_%d.png"%free_var)

plt.savefig('full_freqs.png', dpi=500)

    # print(*freqs, sep='\n')
