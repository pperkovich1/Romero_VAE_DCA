import torch
import numpy as np


class VAE(torch.nn.Module):
    def __init__(self, input_length, num_hidden, num_latent, activation_func, device):
        super(VAE, self).__init__()
        
        ### MISC
        if not type(num_hidden) is list:
            num_hidden = [num_hidden]
        nums = [input_length, *num_hidden, num_latent]
        self.activation_func = activation_func
        self.device = device

        ### ENCODER
        self.hidden_in = torch.nn.ModuleList([torch.nn.Linear(nums[i], nums[i+1]).to(self.device)
                                  for i in range(len(nums)-2)]).to(self.device)
        self.z_mean = torch.nn.Linear(nums[-2], num_latent).to(self.device)
        self.z_log_var = torch.nn.Linear(nums[-2], num_latent).to(self.device)
        
        
        ### DECODER
        self.hidden_out = torch.nn.ModuleList([torch.nn.Linear(nums[i-1], nums[i-2])
                                  for i in range(len(nums),2, -1)]).to(self.device)
        self.linear_4 = torch.nn.Linear(nums[1], input_length).to(self.device)

    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def encoder(self, input_images):
        x = input_images
        (len(x[0])/21)
        for layer in self.hidden_in:
            x=self.activation_func(layer(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded
    
    def decoder(self, encoded):
        x = encoded
        for layer in self.hidden_out:
            x = self.activation_func(layer(x))
        x = self.linear_4(x)
        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, input_images):
        
        z_mean, z_log_var, encoded = self.encoder(input_images)
        decoded = self.decoder(encoded)
        
        return z_mean, z_log_var, encoded, decoded


class CnnVae1D(torch.nn.Module):
    """ Model class for a 1D convolutional variational autoencoder.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, activation_func=torch.nn.Sigmoid(), device='cpu'):
        super(CnnVae1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_func = activation_func
        self.device = device
        #Encoder layers
        self.enc1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding).to(self.device)
        self.z_mean = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding).to(self.device)
        self.z_log_var = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding).to(self.device)
        #Decoder layers
        self.decode1 = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding).to(self.device)
        self.decode2 = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding).to(self.device)
        
        return None
    
    def reparameterize(self, z_mu, z_log_var):
        # Sample epsilon from standard normal distribution
        eps = torch.randn(z_mu.size()).to(self.device)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    
    def encoder(self, x_in):
        """
        """
        x = x_in
        x = self.activation_func(self.enc1(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded
    
    def decoder(self, encoded):
        """
        """
        x = encoded
        d1 = self.activation_func(self.decode1(x))
        d2_out = self.activation_func(self.decode2(d1))
        return d2_out
        
        
    def forward(self, x_in):
        x = x_in
        z_mean, z_log_var, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded
        
    
    def get_dim_out(l_in, layer):
        """Calculates 1D convolution output dimension given the layer and the input dimension of the vector.
        Uses the layer's properties and the formula can be found in the pytorch documentation for Conv1D:
        https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
        """
        return np.floor(((10 + (2*layer.padding[0]) - layer.dilation[0]*(layer.kernel_size[0] - 1) - 1)/layer.stride[0]) + 1)

    def get_transpose_dim_out(l_in, t_layer):
        """Calculates transposed 1D convolution output dimension given the layer and the input dimension
        of the vector.Uses the layer's properties and the formula can be found in the pytorch
        documentaton for ConvTranspose1D:
        https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose1d.html
        """
        return ((l_in-1) * t_layer.stride[0]) - (2*t_layer.padding[0]) + (t_layer.dilation[0] * (t_layer.kernel_size[0]-1)) + t_layer.output_padding[0] + 1