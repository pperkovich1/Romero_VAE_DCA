import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
matplotlib.use('Agg')

cwd = os.getcwd()

if not os.path.exists('stats'):
    os.mkdir('stats')

# stats: epoch_loss, train_ident, train_kld, train_bce, test_loss, test_ident, test_kld, test_bce
stats = os.path.join(cwd, 'stats.pkl')
stats = pickle.load(open(stats, 'rb'))
stats = np.array(stats)[0]
stats_labels = ['training_loss', 'training_identity', 'training_kld', 'training_bce',
                'test_loss', 'test_identity', 'test_kld', 'test_bce']
for i in range(len(stats_labels)):
    plt.figure()
    plt.plot(stats[:,i])
    plt.title(stats_labels[i])
    plt.savefig('stats/'+stats_labels[i]+'.png')
    plt.close()

if not os.path.exists('stats/latents'):
    os.mkdir('stats/latents')

# latent: file, seq, recon_seq, log_s, mu, z
latents = os.path.join(cwd, 'latent_results.pkl')
latents = pickle.load(open(latents, 'rb'))
latents = np.array(latents) 

for latent in latents:
    name, seq, recon_seq, LOG_S, MU, z = latent
    name=name[len('test_sequence_dataset/'):]
    # TODO: change vae.py to output S, MU as 1D arrays
    S = np.exp(LOG_S[0])
    MU = MU[0]
    if not os.path.exists('stats/latents/{}'.format(name)):
        os.mkdir('stats/latents/{}'.format(name))

    ys = []

    for i in range(len(S)):
        plt.figure()
        mu, s = MU[i], S[i]
        x = np.linspace(mu-3*s, mu+3*s, 100)
        y = scipy.stats.norm.pdf(x, mu, s)
        ys.append(y)

        plt.plot(x, y) 
        plt.savefig('stats/latents/{}/{}.png'.format(name, i))
        plt.close();

    plt.figure()
    ys = np.transpose(ys)
    x = np.linspace(-1, 1, 100)
    plt.plot(x, ys)
    plt.savefig('stats/latents/{}/{}.png'.format(name, 'stacked'))
    
print(latents[0])
print(latents.shape)
