import pickle
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.stats
from utils import *
from models import *
import torch
matplotlib.use('Agg')
np.set_printoptions(edgeitems=10, linewidth=1000)

# options = [loss, timestamps, latent vectors, diversity, heat  map]
options_count = 5
options = [0, 0, 0, 0, 0, 1] if len(sys.argv)<=options_count else sys.argv[1:] 



cwd = os.path.join(os.getcwd() , '..')
if not os.path.exists(os.path.join(cwd, 'stats')):
    os.mkdir(os.path.join(cwd, 'stats'))

plt.figure()

def sample_from_model(model_file = 'output/best_model.pt', 
                      dimensions = [75180, [400], 10],
                      inputs_folder = 'data',
                      inputs_data= 'testset.csv',
                      inputs_lims = 'lims.pkl',
                      samples_folder = 'stats/samples',
                      sample_num = 10):
    model_file = open(os.path.join(cwd, model_file), 'rb')
    inputs_folder = os.path.join(cwd, inputs_folder)
    inputs_data = open(os.path.join(inputs_folder, inputs_data))
    inputs_lims = pickle.load(open(os.path.join(inputs_folder, inputs_lims), 'rb'))
    samples_folder = os.path.join(cwd, samples_folder)
    #TODO: check that model has a ModuleList and not just a Module for layer 1
    #TODO: replace 75180 with len(input)
    # model = models.VAE_flexible(*dimensions)
    # if model is not a ModuleList:
    model = VAE(75180, 400, 10)
    model.load_state_dict(torch.load(model_file)['model_state_dict']) 
    loader = torch.utils.data.DataLoader(SequenceDataset(inputs_data, where=inputs_folder),
                        shuffle=False, batch_size=1, num_workers=4)

    # apparetnly inputs_data goes to the end of the file???
    inputs_data.seek(0)
    names = pd.read_csv(inputs_data) ['path']
    for i, name in enumerate(names):
        base = os.path.basename(os.path.normpath(name)) 
        names[i] = (base, os.path.join(samples_folder, base))
    if not os.path.exists(samples_folder):
        os.mkdir(samples_folder)

    for seq, name in zip(loader, names):
        base = name[0]
        name = name[1]
        seq = torch.flatten(seq)
        allowed = [num2aa]*(len(seq)//len(num2aa))
        seq_readable = bit2seq(seq, allowed)
        seqs = [0]*(sample_num+1)
        seqs[0] = SeqRecord(Seq(''.join(seq_readable)), id=base)
        #soo, after flattening, return to 2d to input into model
        seq = seq.reshape(1, -1)
        for i in range(sample_num):
            recon = model(seq)
            # Recon is a tuple, which is weird but oh well
            recon = torch.flatten(recon[0])
            recon = binarize_tensor(recon, inputs_lims)
            recon  = bit2seq(recon, allowed)
            seqs[i+1] = SeqRecord(Seq(''.join(recon)), id=base+'_'+str(i))
        seq_file = open(os.path.join(samples_folder, base), 'w')
        SeqIO.write(seqs, seq_file, format='fasta')
        


        




    '''
    for sequence in inputs:
        binarize_tensor(model(sequence))
    '''


def graph_loss(stats):
    '''
    Input: stats.pkl, 
    Graphs loss over time
    '''
    print('Creating loss graphs')

    # global stats
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

def calc_runtimes(times):
    '''
    Input: file of timestamps
    Calculates average runtime of each section fo code
    '''

    print('Calculating run times')
    
    # global times
    time_file = open('stats/time_data.txt', 'a')
    time_file.write('~~~~~~~~~~~~~\n') # makes it easier to read when running multiple times
    times = pickle.load(open(times, 'rb'))
    times = np.array(times)
    output = {}
    # collects all labels
    for label, time in times[0]:
        if not label in output.keys():
            output[label] = []
            # output.update((label, [])) 

    # converts real time to change in time
    for timestamps in times:
        for i in range(len(timestamps)-1, 0, -1):
            timestamps[i][1] = timestamps[i][1].astype(np.float)-timestamps[i-1][1].astype(np.float) 

    # times = np.moveaxis(times, 0, -1)
    flag = 1
    for timediffs in times:
        for label, timediff in timediffs:
            output[label].append(timediff.astype(np.float))
            if flag:
                print(timediff)
                print(output[label])
                flag = 0

    for label, timediffs in output.items():
        average = np.mean(timediffs)
        output = 'Average {}:'.format(label).ljust(30) + '{:10.6f}\n'.format(average)
        time_file.write(output)
    time_file.close()

def graph_latents(latents):
    '''
    Input: file of latent vectors for each input
    Graphs latent vectors for each input sequence
    '''
    print('Graphing latent vectors')

    # global latents
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

def graph_pairwise(samples):
    '''
    Input: [[probe], [epoch 1 samples], [epoch 2], ...]
    Graphs pairwise identity over time
    '''
    print('Graphing pairwise identity over time')

    # global samples
    samples = pickle.load(open(samples, 'rb'))
    probe = samples[0][0].numpy()
    pos_num = len(probe)//21
    probe = np.reshape(probe, (pos_num, 21))
    probe = im2seq(probe)
    samples = samples[1:]

    averages = []
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

#@TODO: add labels/titles, also legend
def graph_sample_heatmap(sample_dir):
    '''
    Input: directory of output sequences (not one-hot encoded)
    plots amino acid frequency in each position
    '''
    print('Generating heat map')

    # global sample_dir
    sample_dir = os.scandir(sample_dir)

    #@TODO: make figure wider for when I do Big Sequences
    fig = plt.figure() 
    for sample in sample_dir:
        sample = sample.path
        if sample.endswith('.fasta'):
            sample = SeqIO.parse(sample, 'fasta')
            input_seq = np.array(next(sample))
            input_seq = seq2im(input_seq, flatten=False)
            recons = np.array(list(sample))
            recons = np.array([seq2im(seq, flatten=False) for seq in recons])

            recons_sum = np.sum(recons, axis=0)
            # recons_sum = recons_sum/np.linalg.norm(recons_sum,axis=0)
            recons_sum = recons_sum/len(recons)
            recons_sum = np.nan_to_num(recons_sum)
            recons_sum = recons_sum.transpose()
            input_seq = input_seq.transpose()
            masked_sum = ma.array(recons_sum, mask=input_seq)

            heatmap = ma.filled(masked_sum, fill_value=1)
            plt.imshow(heatmap)
            plt.savefig('stats/heatmap.png')
            plt.cla()

            variance = [np.sum(masked_sum, axis=0)]
            plt.imshow(variance)
            plt.savefig('stats/variance.png')
            plt.cla()
    plt.clf()

if __name__ == '__main__':
    ''' expected format of samples.fasta:
    >Input
    >Recon_1
    >Recon_2
    >Recon_3
    [...]
    '''

    if int(options[0]):
        stats = os.path.join(cwd, 'stats.pkl')
        graph_loss(stats)
    if int(options[1]):
        times = os.path.join(cwd, 'times.pkl')
        calc_runtimes(times)
    if int(options[2]):
        latents = os.path.join(cwd, 'latent_results.pkl')
        graph_latents(latents)
    if int(options[3]):
        samples = os.path.join(cwd, 'diversity.pkl')
        graph_pairwise(samples)
    if int(options[4]):
        sample_dir = os.path.join(cwd, 'samples')
        graph_sample_heatmap(sample_dir)
    if int(options[5]):
        sample_from_model()
