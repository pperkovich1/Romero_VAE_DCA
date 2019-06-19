import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
matplotlib.use('Agg')

cwd = os.getcwd()


options = {'loss': False, 'times': True, 'latents': False}

if options['loss']:
    if not os.path.exists('stats'):
        os.mkdir('stats')

    stats = os.path.join(cwd, 'stats.pkl')
    stats = pickle.load(open(stats, 'rb'))
    stats = np.array(stats)[0]
    stats_labels = ['epoch_loss', 'train_ident', 'train_kld', 'train_bce',
                    'test_loss', 'test_ident', 'test_kld', 'test_bce']
    for stat in stats_labels:
        plt.figure()
        plt.plot([data[stat] for data in stats])
        plt.title(stats_labels[i])
        plt.savefig('stats/'+stat+'.png')
        plt.close()

if options['times']:
    time_file = open('time_data.txt', 'a')
    time_file.write('~~~~~~~~~~~~~\n') # makes it easier to read when running multiple times
    times = os.path.join(cwd, 'times.pkl')
    times = pickle.load(open(times, 'rb'))
    times = np.array(times)[0]
    for timestamps in times:
        for i in range(len(timestamps)-1, 0, -1):
            timestamps[i][1] = timestamps[i][1].astype(np.float)-timestamps[i-1][1].astype(np.float)
    times = np.moveaxis(times, 0, -1)
    for timestamp in times[1:]:
        label = timestamp[0][0]
        average = np.nanmean(timestamp[1].astype(np.float))
        output = 'Average {}:'.format(label).ljust(30) + '{:10.6f}\n'.format(average)
        print(output)
        time_file.write(output)
    time_file.close()
    
    
 
if options['latents']:
    if not os.path.exists('stats/latents'):
        os.mkdir('stats/latents')

    # latent: file, seq, recon_seq, log_s, mu, z
    latents = os.path.join(cwd, 'latent_results.pkl')
    latents = pickle.load(open(latents, 'rb'))
    latents = np.array(latents) 

    for latent in latents:
        name, seq, recon_seq, LOG_S, MU, z = latent
        name = name[len('test_sequence_dataset/'):len('.fasta')]
        name = name.replace('/', '')
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
