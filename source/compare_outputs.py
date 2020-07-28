import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

import utils
from read_config import Config
import examine_model

config = Config('../config_1000_15.yaml')
dataset = utils.get_dataset_from_config(config)
model = utils.load_model_from_config(config)
loader = DataLoader(dataset=dataset, batch_size = 100)
device = config.device

outputs = []
with torch.no_grad():
    for batch_id, (input_images, weights) in enumerate(loader):
        input_images = input_images.to(device)
        weights = weights.to(device)
        z_mean, z_log_var, encoded, recon_images = model(input_images)
        outputs.append(recon_images)
        break

outputs = outputs[0]
outputs = utils.softmax(outputs)
print(outputs)
print(outputs.size())

