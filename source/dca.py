"""
    Implementation of DCA (Direct Coupling Analysis) model.  DCA is a special
    case of VAEs with no hidden layers.  The output layer is directly connected
    to the input layer with an almost dense layer. 

    This pytorch implementation shall closely mimic sokrypton's seqmodel
    implementation by Sergey Ovchinnikov.
    https://github.com/sokrypton/seqmodels/blob/master/seqmodels.ipynb

"""

import numpy as np # Used to import weights only
import pandas as pd # Used to return the scores
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

        # These are the real weights of the model. Although we do the gradient 
        # descent step on the w we need to use the symmetrized weights
        # self.weights to get the coupling values e_ij(a,b)
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

    def plot_loss_curve(losses, annotatation_str="", save_fig_path=None, 
            model_name=""):
        """ Save graph of loss curves 
            FIXME: This function is quite generic and should live in utils or
            somewhere else so that it can be merged with the VAE model plotting
            code as well.
        """
        import matplotlib.pyplot as plt # hide this import here so we don't pollute
        plt.plot(losses, "o-")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve :{model_name}")
        bbox = dict(boxstyle="round", fc="0.8")
        plt.annotate(annotatation_str, (0.5, 0.5), xycoords='axes fraction',
                    bbox=bbox);
        if save_fig_path is not None:
            plt.savefig(save_fig_path)

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
    train_step = DCA.make_train_step(model, DCA.create_loss_function(), optimizer)

    losses = []
    for epoch in range(num_epochs):
        loss = train_step(x=msa, x_weights=msa_weights, x_cat=msa_cat)
        losses.append(loss)
        if verbose:
            print(f"Epoch: {epoch:02d} Loss={loss:.2f}")
    if ret_losses_only:
        ret = losses
    else:
        ret = {'losses':losses, 'model':model, 'optimizer':optimizer,
                'weights': model.weights.detach().numpy(),
                'bias': model.bias.detach().numpy()
                }
    return ret

def calc_contact_score_from_weights(weights, do_apc=True, as_pandas=True, 
        dist_greater=5, sort=True):
    """Calculate the contact scores from the weights array

    The equation numbers below correspond to 2013 Ekeberg: Improved contact
    prediction in proteins: Using pseudo-likelihoods to infer Potts models

    Args:
        weights:        np array dims (L,q,L,q) representing coupling weights
        do_apc:         whether to do APC correction or not. 
        as_pandas:      whether to convert matrix of scores into a dataframe
        dist_greater:   is only applied if as_pandas is True and it returns
                            indices i, j where i < j and |i-j| > dist_greater
        sort:           is only applied if as_pandas is True

    """
    # convert to zero-sum guage     
    jprime = weights - np.mean(weights, axis=0, keepdims=True) \
                 - np.mean(weights, axis=2, keepdims=True) \
                 + np.mean(weights, axis=(0,2), keepdims=True)
    # Frobenius Norm (Eqn 27)
    fn = np.linalg.norm(jprime, axis=(1,3), ord='fro')

    # Corrected Norm (Eqn 28)
    cn = fn
    if do_apc:
        cn = fn - (np.mean(fn, axis=0, keepdims=True) 
                    * np.mean(fn, axis=1, keepdims=True) 
                    / np.mean(fn, axis=(0,1), keepdims=True))
    ret = cn
    if as_pandas:
        i, j = np.triu_indices_from(cn)
        df = pd.DataFrame({'i':i, 'j':j, 'score':cn[i,j]})
        df = df[(df.i - df.j).abs() > dist_greater] 
        if sort:
            df = df.sort_values('score', ascending=False)
            df = df.reset_index(drop=True)
        ret = df
    return ret
        

if __name__ == "__main__":
    import time
    import argparse
    import read_config
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename",
                    help="input config file in yaml format")
    args = parser.parse_args()

    config = read_config.Config(args.config_filename)

    start_time = time.time()
    msa, msa_weights = load_full_msa_with_weights(
            msa_path=config.aligned_msa_fullpath,
            weights_path=config.weights_fullpath)
    ret = train_dca_model(device=config.device,
                       msa=msa, msa_weights=msa_weights,
                       learning_rate = config.learning_rate,
                       num_epochs=config.epochs)
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    # save parameters
    
    with open(config.dca_params_fullpath, 'wb') as fh:
        pickle.dump({k:ret[k] for k in ["weights", "bias"]}, fh) 

    # plot loss curve
    DCA.plot_loss_curve(losses=ret['losses'],  
            annotatation_str = str(ret['optimizer']),
            save_fig_path = config.lossgraph_fullpath,
            model_name= config.model_name)

    # save loss curve data
    with open(config.loss_fullpath, 'wb') as fh:
        pickle.dump(ret['losses'], fh) 

    # save model state
    torch.save(ret['model'].state_dict(), config.model_fullpath)


     

