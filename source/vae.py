import torch
import time
import os

#local files
import config
from model import VAE
from dataloader import MSADataset

def train_model(model, trainloader, valloader, max_epochs, convergence_limit):
	start_time = time.time()
	min_loss = 999999
	no_improvement = 0
	train_loss = []
	test_loss = []
	for epoch in range(max_epochs):
		print('Epoch: %i\tTime elapsed:%.2f min'%(epoch, (time.time()-start_time)))
		train_loss.append([])
		test_loss.append([])

		if no_improvement > convergence_limit:
			print("convergence at %i iterations" % epoch)
			break
		for batch_id, (input_images, weights) in enumerate(trainloader):
			input_images = input_images.to(device)
			weights = weights.to(device)

			z_mean, z_log_var, encoded, recon_images = model(input_images)
			kld = utils.kl_divergence(z_mean, z_log_var)
			bce = utils.bce(recon_images, input_images, weights)
			loss = kld+bce

			optimizer.zero_grad()
			loss.backward
			optimizer.step()

			train_loss[epoch].append(loss)

		#validation
		with torch.no_grad():
			for batch_id, (input_images, weights) in enumerate(valloader):
				input_images = input_images.to(device)
				weights = weights.to(device)

				z_mean, z_log_var, encoded, recon_images = model(input_images)
				kld = utils.kl_divergence(z_mean, z_log_var)
				bce = utils.bce(recon_images, input_images, weights)
				loss = kld+bce
				if min_loss < loss:
					no_improvement += 1
				else:
					min_loss = loss
					no_improvement = 0

				test_loss[epoch].append(loss)
	torch.save(model.state_dict(), "model.pt")

def main():
	input_length = config.input_length
	num_hidden = config.num_hidden
	num_latent = config.num_latent
	activation_func = config.activation_func
	model = VAE(input_length, num_hidden, num_latent, activation_func)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if os.path.exists(config.prev_model):
		print("Loading saved model...")
		model.load_state_dict(torch.load(config.prev_model))

	model.to(device)

	dataloader = MSADataset(config.msa)
	trainsize = int(len(dataloader)*.9)
	valsize = len(dataloader) - trainsize
	(trainloader, valloader) = torch.utils.data.random_split(dataloader, (trainsize, valsize))

	max_epochs = config.max_epochs
	convergence = config.convergence
	train_model(model, trainloader, valloader, max_epochs, convergence_limit)

if __name__=='__main__':
	main()
