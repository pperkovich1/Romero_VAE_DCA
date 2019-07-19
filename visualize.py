import pickle
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from utils import *
from models import *
import torch
matplotlib.use('Agg')

# options = [loss, timestamps, latent vectors, diversity, heat  map]
options_count = 5
options = [1, 1, 1, 1, 1] if len(sys.argv)<=options_count else sys.argv[1:] 

cwd = os.path.join(os.getcwd()) # , 'results') 
stats = os.path.join(cwd, 'stats.pkl')
times = os.path.join(cwd, 'times.pkl')
latents = os.path.join(cwd, 'latent_results.pkl')
samples = os.path.join(cwd, 'diversity.pkl')

model_data = os.path.join(cwd, 'best_model.pt')
size=75180
latent_dim=250
hidden_sizes=[250]
batch_size = 100

if not os.path.exists('stats'):
    os.mkdir('stats') 
plt.figure()

def graph_loss():
    print('Creating loss graphs') 
    stats = pickle.load(open(stats, 'rb'))
    stats = np.array(stats)
    stats_labels = ['epoch_loss', 'train_ident', 'train_kld', 'train_bce',
                    'test_loss', 'test_ident', 'test_kld', 'test_bce']
    for stat in stats_labels:
        plt.clf()
        # plt.figure()
        plt.plot([data[stat] for data in stats])
        plt.title(stat)
        plt.savefig('stats/'+stat+'.png')
        # plt.close()

def calc_runtimes():
    print('Calculating run times')

    time_file = open('stats/time_data.txt', 'a')
    time_file.write('~~~~~~~~~~~~~\n') # makes it easier to read when running multiple times
    times = pickle.load(open(times, 'rb'))
    times = np.array(times)
    for timestamps in times:
        for i in range(len(timestamps)-1, 0, -1):
            timestamps[i][1] = timestamps[i][1].astype(np.float)-timestamps[i-1][1].astype(np.float)
    times = np.moveaxis(times, 0, -1)
    for timestamp in times:
        label = timestamp[0][0]
        average = np.nanmean(timestamp[1].astype(np.float))
        output = 'Average {}:'.format(label).ljust(30) + '{:10.6f}\n'.format(average)
        time_file.write(output)
    time_file.close()

def graph_latents():
    print('Graphing latent vectors')

    if not os.path.exists('stats/latents'):
        os.mkdir('stats/latents')

    # latent: file, seq, recon_seq, log_s, mu, z
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
            plt.clf()
            mu, s = MU[i], S[i]
            x = np.linspace(mu-3*s, mu+3*s, 100)
            y = scipy.stats.norm.pdf(x, mu, s)
            ys.append(y)
        
        plt.clf()
        plt.plot(x, y) 
        plt.savefig('stats/latents/{}/{}.png'.format(name, i))

    plt.clf()
    ys = np.transpose(ys)
    x = np.linspace(-1, 1, 100)
    plt.plot(x, ys)
    plt.savefig('stats/latents/{}/{}.png'.format(name, 'stacked')) 

def graph_pairwise(): 
    print('Graphing pairwise identity over time')

    samples = torch.load(samples, map_location='cpu')
#    samples = pickle.load(open(samples, 'rb'))
    probe = samples[0][0].numpy()
    pos_num = len(probe)//21
    probe = np.reshape(probe, (pos_num, 21))
    probe = im2seq(probe)
    samples = samples[1:]

    averages= []
    for sample in samples:
        # sample = [im2seq(np.reshape(binarize_image(seq), (pos_num, 21))) for seq in sample]
        average = 0
        for seq in sample:
            seq = seq.numpy()
            seq = np.reshape(seq, (pos_num, 21))
            seq = binarize_image(seq)
            seq = im2seq(seq)
            average += identity(probe, seq)
        average = average*21/len(sample)
        averages.append(average)
    plt.clf()
    plt.plot(averages)
    plt.title('diversity over time')
    plt.savefig('stats/diversity.png')
    plt.close()

def graph_sample_heatmap():
    with torch.no_grad():
        print('Generating heatmap of samples')
        sampleAmount = 1000
        cpus = os.cpu_count()
        lims = pickle.load(open('lims.pkl', 'rb'))

        model_data = torch.load(globals()['model_data'], map_location='cpu')
        model_state_dict = model_data['model_state_dict']
        model = VAE_flexible(l=size, latent_size=latent_dim, hidden_sizes=hidden_sizes)
        model.load_state_dict(model_state_dict)

        seqs = pd.read_csv('seqDbPaths.csv')
        seqs = seqs.head(1)
        seqs.to_csv('fullset.csv')
        seqloader = DataLoader(SequenceDataset('fullset.csv'), shuffle=True,
                               batch_size=batch_size, num_workers=cpus)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for seq in seqloader:
            model.encode(seq)
            recon_sum = torch.from_numpy(np.zeros_like(seq))
            for i in range(sampleAmount):
                recon_seq = binarize_tensor(model.decode()[0], lims)
                recon_sum += recon_seq 
            recon_sum = recon_sum.numpy().reshape(20, -1)
            print(recon_sum[:,:15]*1000)
            plt.figure()
            plt.imshow(recon_sum[:,:15]*1000)
            plt.savefig('stats/heatmap.png')
            
        
        print('Didn\'t crash!')

    
if int(options[0]):
    graph_loss() 
if int(options[1]):
    calc_runtimes() 
if int(options[2]):
    graph_latents()
if int(options[3]):
    graph_pairwise()
if int(options[4]):
    graph_sample_heatmap()
