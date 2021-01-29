import torch.nn.functional as F
import torch

def get_input_length(dataset):
    sample = dataset.__getitem__(0)[0]
    return len(sample)


