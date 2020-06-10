"""
    Implementation of DCA (Direct Coupling Analysis) model.  DCA is a special
    case of VAEs with no hidden layers.  The output layer is directly connected
    to the input layer with an almost dense layer. 

    This pytorch implementation shall closely mimic sokrypton's seqmodel
    implementation by Sergey  Ovchinnikov
    https://github.com/sokrypton/seqmodels/blob/master/seqmodels.ipynb

"""

import numpy as np # Used to import weights only
import torch
from dataloader import MSADataset, OneHotTransform

class DCA(torch.nn.Module):

    """Pseudo-likelihood method to estimate maximum entropy probability dist"""

    def __init__(self, ncol, ncat, Neff, lam_w=0.01, lam_b=0.01, b_ini=None):
        super().__init__()
        # weights
        self.w = torch.nn.Parameter(torch.zeros((ncol, ncat, ncol, ncat),
                        dtype=torch.float, requires_grad=True))
        # Variable to store symmetric weights. They are used in the loss function
        # so they are saved here and recomputed each training step
        self.weights = None
        # weights_eye is used to remove weights between (i,a) and (i,b)
        self.weights_eye = torch.reshape(1 - torch.eye(ncol), (ncol,1,ncol, 1))

        # biases
        if b_ini is None:
            b_ini = torch.zeros((ncol, ncat), dtype=torch.float)
        self.bias = torch.nn.Parameter(b_ini.clone().detach().requires_grad_(True))

        self.Neff = Neff
        self.lam_w = lam_w
        self.lam_b = lam_b
        self.ncol = ncol # required to compute regularization loss
        self.ncat = ncat # not used but saved anyway


    def forward(self, x):
        """Predictions are going to be MSA logits"""
        x_msa = x
        # we do not want weights between the various nodes in a given position.
        # i.e. weights between nodes (i, a) and (j, b) only exist if i not = j
        # so set these weights to zero
        w_eye = self.w * self.weights_eye
        # symmetrize w so that the weight between (i,a) and (j, b) is the
        # same as the weight between (j, b) and (i, a)

        # Why does sokrypton not divide by 2 here?
        #       Not sure but here is a possible explanation....
        #       There are certain transformations of weights and biases that
        #       change the intractable normalizing constant (Z) in a way that
        #       gives the same probability distribution. (Gauage Invariance)
        #       However, regularization fixes a guage so this could just be an
        #       oversight.
        self.weights = w_eye + w_eye.permute(2,3,0,1)
        x_logit = torch.tensordot(x_msa, self.weights, 2) + self.bias
        return x_logit

    def calc_reg_w(self):
        return self.lam_w * \
                    torch.sum(torch.mul(self.weights, self.weights)) * \
                    0.5 * (self.ncol-1) * 20.0

    def calc_reg_b(self):
        return self.lam_b * torch.sum(torch.mul(self.bias, self.bias))

    def create_dca_model(msa, msa_weights, *args, **kwargs):
        """Factory function to create a model with a pseudocount bias term"""

        with torch.no_grad():
            # Number of effective sequences
            Neff = torch.sum(msa_weights)

            nseq, ncol, ncat = tuple(msa.shape)

            # start bias with b_ini instead of zeros
            pseudo_count = 0.01 * torch.log(Neff)
            b_ini = torch.log(torch.sum(msa.T * msa_weights, axis=-1).T +
                    pseudo_count)
            b_ini = b_ini - torch.mean(b_ini, -1, keepdim=True)

            return DCA(ncol=ncol, ncat=ncat, Neff=Neff, b_ini=b_ini, 
                                *args, **kwargs)


def create_loss_function():
    """ Create a function that will compute cross entropy loss """
    ce_loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    
    def calc_loss(x_logit, x_cat, x_weights, model):
        loss_ce = ce_loss_func(x_logit.permute(0,2,1), x_cat)
        loss_ce = loss_ce.sum(dim=-1)
        loss_ce = (loss_ce * x_weights).sum()

        reg = model.calc_reg_w() + model.calc_reg_b()
    
        loss = (loss_ce + reg) / model.Neff
        return loss
    return calc_loss

def make_train_step(model, loss_fn, optimizer):
    """Builds function that performs a step in the train loop """
    def train_step(x, x_weights, x_cat):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        x_logit = model(x)
        # Computes loss
        loss = loss_fn(x_logit, x_cat, x_weights, model)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step


def load_full_msa_with_weights(msa_path, weights_path=None, verbose=True):

    weights = None
    if weights_path is None:
        print(f"weights_path is none. Setting all weights to 1.")
    else:
        print(f"Reading Weights from {str(weights_path)}")
        weights = np.load(weights_path)

    dataset = MSADataset(msa_path, weights=weights,
                     transform=OneHotTransform(21, flatten=False))
    msa = torch.utils.data.DataLoader(dataset, len(dataset))
    
    # only load the first element of the dataset enumerator
    # (this is the entire dataset)
    for _, msa_data in enumerate(msa):
        msa  = msa_data[0]
        msa_weights = msa_data[1]
        break
    
    if verbose:
        print(f"Data.shape = {msa.shape}")
        print(f"Weights.shape = {msa_weights.shape}")
    return msa, msa_weights

def train_dca_model(device, msa, msa_weights, num_epochs, learning_rate, 
        verbose=True, ret_losses_only=False):
    # MSA one-hot and large so don't want to make copies unless necessary
    msa = msa.to(device) 
    msa_weights = msa_weights.to(device)
    msa_cat = msa.argmax(dim=2) # type LongTensor

    model = DCA.create_dca_model(msa, msa_weights)
    model.to(device)

    # Tell the optimizer which weights we want to update
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_step = make_train_step(model, create_loss_function(), optimizer)

    losses = []
    for epoch in range(num_epochs):
        loss = train_step(x=msa, x_weights=msa_weights, x_cat=msa_cat)
        losses.append(loss)
        if verbose:
            print(f"Epoch: {epoch:02d} Loss={loss:.2f}")
    if ret_losses_only:
        ret = losses
    else:
        ret = {'losses':losses, 'model':model, 'optimizer':optimizer}
    return ret

if __name__ == "__main__":
    pass
