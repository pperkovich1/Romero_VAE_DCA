"""
    Implementation of DCA (Direct Coupling Analysis) model.
    DCA is a special case of VAEs with no hidden layers.  The output layer is
    directly connected to the input layer with an almost dense layer. 

    This pytorch implementation shall closely mimic sokrypton's seqmodel
    implementation by Sergey  Ovchinnikov
    https://github.com/sokrypton/seqmodels/blob/master/seqmodels.ipynb

"""

import torch

class DCA(torch.nn.Module):

    def __init__(self, input_length):
        pass
