''' Author: Juan R. Diaz Rodriguez, James L. Wang
last updated: 2019-06-18 JLW
'''

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from models import *
import sys
import pickle
import timeit
from timeit import default_timer as timer


def main():
    # I was having a weird cuda problem on my local machine,
    # vvv was solution posted on https://github.com/pytorch/pytorch/issues/20990
    torch.cuda.current_device()
    # if true, trains on full dataset
    full_run = True

    cpus = os.cpu_count()
    # Input order:    layers, batch size, learning rate, epochs, convergence
    # then all of the hidden dimensions
    # typical values: 1       100         .05              100     .1           400  20
    num_layers = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    num_epochs = int(sys.argv[4])
    convergence = float(sys.argv[5])
    h_dims = []
    for i in range(6, 6+num_layers, 1):
        h_dims.append(int(sys.argv[i]))
    latent_dim = int(sys.argv[6+num_layers])
    improve_step_pct = .1


    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Batch Size =', batch_size)
    print('Learning Rate =', learning_rate)
    print('Total Epochs = ', num_epochs)
    print('Convergence = ', convergence)
    print('Hidden Dim = ', h_dims)
    print('Latent Dim = ', latent_dim)
    #separate path DB into train/test dbs for dataloader
    allseqpaths = pd.read_csv('seqDbPaths.csv') 
    if full_run:
        allseqpaths = allseqpaths['path']
    else:
        allseqpaths = allseqpaths['path'].head(40)
    train, test, out_train, out_test = train_test_split(allseqpaths, allseqpaths, test_size=0.4)
    trainpaths = pd.DataFrame(columns=['path'])
    # make dbs for dataloader
    trainpaths['path'] = train
    trainpaths.to_csv('trainset.csv')
    testpaths = pd.DataFrame(columns=['path'])
    testpaths['path'] = test
    testpaths.to_csv('testset.csv')
    lims = pickle.load(open('lims.pkl', 'rb'))


    #Initialize Data Loaders
    trainloader = DataLoader(SequenceDataset('trainset.csv'),  shuffle=True, batch_size=batch_size, num_workers=cpus)
    testloader = DataLoader(SequenceDataset('testset.csv'),  shuffle=True, batch_size=batch_size, num_workers=cpus)

    # check if cuda is avilable, otherwise set device to cpu.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Size of input vector for VAE initialization
    size = len(read_sequence(os.path.join(os.getcwd(), allseqpaths[0])))
    pos_num = int(size/21)

    # Initialize vae and start training
    # if num_layers==1:
    # model = VAE(l=size, latent_size=latent_dim, hidden_size=h_dims[0], device=device).to(device)
    # elif num_layers==2:
    #     model = VAE_double(l=size, latent_size=latent_dim, hidden_size_1=h_dims[0], hidden_size_2=h_dims[1]).to(device)
    # else
    model = VAE_flexible(l=size, latent_size=latent_dim, hidden_sizes=h_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loss = []
    count = 0
    min_loss = 999999999
    best_iter = 0

    # network evaluation stuff
    stats = []
    times = []
    probe = iter(testloader).__next__()[0:1] # grabs a tensor from testset
    probe = probe.to(device)
    probe_amount = 50
    probe_results = [probe]

    cwd = os.getcwd()
    # folder = os.path.join(cwd, 'results')
    # if(not os.path.exists(folder)): # added to make repeated testing on personal machine a bit easier
    # 	os.mkdir(os.path.join(cwd,folder))

    for epoch in range(num_epochs):
        print(epoch)
        if count >= (convergence*num_epochs):
                print("Convergence at %i iterations." % epoch)
                print("Minimum loss at %i iterations." % best_iter)
                break
        else:
            timestamps=[]
            #train
            epoch_loss = []
            train_kld = []
            train_bce = []
            train_ident = []
            for seq in trainloader:
                timestamps.append(['batch_start', timer()]) # batch start time
                trainseq = seq.float().to(device)
                optimizer.zero_grad()
                timestamps.append(['model_preparation', timer()])
                recon_seq, s, mu = model(trainseq)
                timestamps.append(['forward_pass', timer()])
                bce = F.binary_cross_entropy(trainseq.float(),
                                            recon_seq.detach().float(),
                                            reduction='sum')
                kld = kl_divergence(mu, s)
                l = bce + kld
                timestamps.append(['train_loss', timer()])
                l.backward()
                optimizer.step()
                timestamps.append(['back_prop', timer()])
                epoch_loss.append(float(l.item()/len(trainseq)))
                train_kld.append(float(kld.item()/len(trainseq)))
                train_bce.append(float(bce.item()/len(trainseq)))
                 # calculate identity
                if len(trainseq) > 1: #if batching
                    for i in range(len(trainseq)):
                        train_ident.append(tensor_pairwise_identity(trainseq[i], recon_seq[i], lims))
                else: # no batch
                    train_ident.append(tensor_pairwise_identity(trainseq[i], recon_seq[i], lims))
                timestamps.append(['train_ident', timer()])
            train_ident = np.mean(train_ident)
            epoch_loss = np.mean(epoch_loss)
            train_kld = np.mean(train_kld)
            train_bce = np.mean(train_bce)
            timestamps.append(['train_mean', timer()])
                                  
            #test
            with torch.no_grad():
                test_loss = []
                test_kld = []
                test_bce = []
                test_ident = []
                for seqt in testloader:
                    timestamps.append(['test_start', timer()])
                    testseq = seqt.float().to(device)
                    recon_seq, s, mu = model(testseq)
                    bce = F.binary_cross_entropy(testseq.float(),
                                            recon_seq.detach().float(),
                                            reduction='sum')
                    kld = kl_divergence(mu, s)
                    l_test = bce + kld
                    test_loss.append(float(l_test.item()/len(testseq)))
                    test_kld.append(float(kld)/len(testseq))
                    test_bce.append(float(bce)/len(testseq))
                    timestamps.append(['test_loss', timer()])
                    # calculate identity
                    if len(testseq) > 1: #if batching
                        for i in range(len(testseq)):
                            test_ident.append(tensor_pairwise_identity(testseq[i], recon_seq[i], lims))
                    else: # no batch
                        test_ident.append(tensor_pairwise_identity(testseq[i], recon_seq[i], lims))
                test_ident = np.mean(test_ident) #TODO: tell Juan about this typo
                test_loss = np.mean(test_loss)
                test_kld = np.mean(test_kld)
                test_bce = np.mean(test_bce)
                timestamps.append(['test_mean', timer()])
                probe_results.append([])
                for i in range(probe_amount):
                    probe_results[-1].append(model(probe)[0])
                timestamps.append(['sample_diversity', timer()])

                if min_loss < (test_loss - test_loss*improve_step_pct):
                    count += 1 # add no-improvement count. 
                    timestamps.append(['no_save', np.NaN])
                elif full_run:
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_loss': test_loss,
                    'test_kld': test_kld,
                    'test_bce': test_bce,
                    'train_loss': epoch_loss,
                    'train_kld': train_kld,
                    'train_bce': test_kld,
                    'test_identity':test_ident,
                    'train_identity':train_ident,
                    },
                    'best_model.pt')
                    min_loss = test_loss
                    best_iter = epoch
                    count = 0 # reset no-improvement count
                    timestamps.append(['save_model', timer()])
                
                
                # ===================log========================
                print('iter = %i\ttrain loss = %0.4f\ttest loss = %0.4f' % (epoch, l.item(), test_loss.item()))
                stats.append({'epoch_loss':epoch_loss,'train_ident':train_ident,'train_kld':train_kld,'train_bce':train_bce,
                              'test_loss':test_loss,'test_ident':test_ident,'test_kld':test_kld,'test_bce':test_bce})
                timestamps = np.array(timestamps)
                times.append(timestamps)
                print('Timings:\n', timestamps)

    pickle.dump(stats, open('stats.pkl', 'wb'))
    pickle.dump(times, open('times.pkl', 'wb'))
    torch.save(probe_results, open('diversity.pkl', 'wb'))
    # pickle.dump(probe_results, open('results/diversity.pkl', 'wb'))


    # store all latest space variables for all sequences in dataset.
    latent_results = []
    for seqpath in allseqpaths:
        seq = np.array(list(SeqIO.parse(open(seqpath), 'fasta'))[0].seq, dtype=np.float)
        seq = torch.tensor([seq]).float().to(device)
        recon_seq, s, mu = model(seq)
        
        latent_results.append([
            seqpath,
            seq.data.cpu().numpy(),
            recon_seq.data.cpu().numpy().flatten(),
            s.data.cpu().numpy(),
            mu.data.cpu().numpy(),
            model.z.data.cpu().numpy()])
    pickle.dump(latent_results,open('latent_results.pkl', 'wb')) 

if __name__ == '__main__':
    main()
    print('Runtime (s):\t', timeit.timeit('main()', number=1))
