# libraries
import pickle
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# local files
import utils
from train_model import load_model_from_config
from dataloader import MSADataset, OneHotTransform
from read_config import Config

config_path = '../config_27.yaml'
config = Config(config_path)
print('uhh idk what to call this')




# dataset = MSADataset(config.aligned_msa_fullpath, transform=OneHotTransform(21))

# input_length = utils.get_input_length(dataset)
# model = load_model_from_config(input_length=input_length, config=config)


# loader = DataLoader(dataset, batch_size = 3)


# with torch.no_grad():
#     sample_size = 100

#     outputs = torch.tensor([])
#     for batch_id, (input_images, weights) in enumerate(loader):
#         outputs = torch.cat((outputs, input_images), 1)
#         for i in range(sample_size):
#             z_mean, z_log_var, encoded, recon_images = model(input_images)
#             outputs = torch.cat((outputs, recon_images), 1)
#         break

#     frequencies = []
#     masks = []
#     for image_clump in outputs:
#         images = torch.reshape(image_clump, (-1, input_length))
#         images = utils.softmax(images)
#         input_image = images[0]
#         recon_images = images[1:]
#         freqs = torch.sum(recon_images, 0)/len(recon_images)

#         input_image = torch.reshape(input_image, (-1, 21))
#         freqs = torch.reshape(freqs, (-1, 21))

#         mask = []
#         for i, row in enumerate(freqs):
#             if len(row.nonzero()) > 1:
#                 mask.append(i)
#         mask = torch.tensor(mask)

#         frequencies.append([input_image, freqs, mask])

#     with open('freqs.pkl', 'wb') as fh:
#         pickle.dump({'frequencies': frequencies}, fh)



with open('freqs.pkl', 'rb') as fh:
    frequencies = pickle.load(fh)['frequencies']


seq_1 = frequencies[0]
input_image_1 = seq_1[0]
freqs_1 = seq_1[1]
mask_1 = seq_1[2]

seq_2 = frequencies[1]
input_image_2 = seq_2[0]
freqs_2 = seq_2[1]
mask_2 = seq_2[2]

seq_3 = frequencies[2]
input_image_3 = seq_3[0]
freqs_3 = seq_3[1]
mask_3 = seq_3[2]



for i in range(len(input_image_1)):
    in_1 = input_image_1[i]
    in_2 = input_image_2[i]
    in_3 = input_image_3[i]

    if (not torch.equal(in_1, in_2)) or (not torch.equal(in_1, in_3)) or (not torch.equal(in_2, in_3)):
        print('position %d:\t%d\t%d\t%d\t'%(i, in_1.nonzero(), in_2.nonzero(), in_3.nonzero()))

# Positions where input sequences disagree


# Print positions where input residue is NEVER selected
# problems = []
# for i, row in enumerate(input_image):
#     val, j = torch.max(row, 0)
#     if freqs[i, j]==0:
#         problems.append(i)
# print(problems)
# print(len(problems))

# Plot only positions with variation in residue
# input_masked = torch.index_select(input_image, 0, mask)
# freqs_masked = torch.index_select(freqs, 0, mask)

# fig, (ax1, ax2) = plt.subplots(ncols=2)

# im1 = ax1.imshow(input_masked, aspect='auto')
# ax1.set_xlabel('original image')

# yticks = [str(i) for i in mask.numpy()]

# ax1.set_yticks(np.arange(len(yticks)))
# ax1.tick_params(axis='y', labelsize=3)
# ax1.set_yticklabels(yticks)
# # for edge, spine in ax1.spines.items():
# #     spine.set_visible(False)

# im2 = ax2.imshow(freqs_masked, aspect='auto')
# ax2.get_yaxis().set_visible(False)
# ax2.set_xlabel('reconstruction frequencies')
# # cbar = ax2.figure.colorbar(im2, ax=ax2)

# plt.savefig('freqs.png', dpi=500)