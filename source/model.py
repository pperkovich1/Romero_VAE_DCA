import torch


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
        self.hidden_in = torch.nn.ModuleList([torch.nn.Linear(nums[i], nums[i+1])
                                  for i in range(len(nums)-2)])
        self.z_mean = torch.nn.Linear(nums[-2], num_latent)
        self.z_log_var = torch.nn.Linear(nums[-2], num_latent)
        
        
        ### DECODER
        self.hidden_out = torch.nn.ModuleList([torch.nn.Linear(nums[i-1], nums[i-2])
                                  for i in range(len(nums),2, -1)])
        self.linear_4 = torch.nn.Linear(nums[1], input_length)

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
