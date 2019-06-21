''' Author: Juan R. Diaz Rodriguez, James L. Wang
last updated: 2019-06-18 JLW
'''

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from models import *
import sys
import pickle
from timeit import default_timer as timer

# if true, trains on full dataset
full_run = False

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


#Initialize Data Loaders
trainloader = DataLoader(SequenceDataset('trainset.csv'),  shuffle=True, batch_size=batch_size, num_workers=cpus)
testloader = DataLoader(SequenceDataset('testset.csv'),  shuffle=True, batch_size=batch_size, num_workers=cpus)

# check if cuda is avilable, otherwise set device to cpu.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Size of input vector for VAE initialization
size = len(read_sequence(os.path.join(os.getcwd(), allseqpaths[0])))
pos_num = int(size/21)

# Initialize vae and start training
if num_layers==1:
    model = VAE(l=size, latent_size=latent_dim, hidden_size=h_dims[0]).to(device)
elif num_layers==2:
    model = VAE_double(l=size, latent_size=latent_dim, hidden_size_1=h_dims[0], hidden_size_2=h_dims[1]).to(device)
else:
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
probe_amount = 50
probe_results = [probe]

cwd = os.getcwd()
folder = os.path.join(cwd, 'results')
if(not os.path.exists(folder)): # added to make repeated testing on personal machine a bit easier
	os.mkdir(os.path.join(cwd,folder))
for epoch in range(num_epochs):
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
            # convert from binary to sequence
            for s1, s2 in zip(trainseq,recon_seq):
                print('~~~~~')
                print(len(s1))
                print(pos_num)
                print(s1)
                s1 = np.reshape(list(s1.cpu().data),(pos_num,21))
                print(s1)
                print(len(s1))
                s2 = np.reshape(list(s2.cpu().data),(pos_num,21))
                s2 = binarize_image(s2)
                s1 = im2seq(s1)
                s2 = im2seq(s2)
                train_ident.append(identity(s1,s2))
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
                # convert from binary to sequence
                for s1, s2 in zip(testseq,recon_seq):
                    s1 = np.reshape(list(s1.cpu().data),(pos_num,21))
                    s2 = np.reshape(list(s2.cpu().data),(pos_num,21))
                    s2 = binarize_image(s2)
                    s1 = im2seq(s1)
                    s2 = im2seq(s2)
                    test_ident.append(identity(s1,s2))
                timestamps.append(['test_ident', timer()])
            # test_ident = np.mean(train_ident) # not train, but test? (vvv)
            test_ident = np.mean(test_ident) # I think this is what is supposed to be here?
            test_loss = np.mean(test_loss)
            test_kld = np.mean(test_kld)
            test_bce = np.mean(test_bce)
            timestamps.append(['test_mean', timer()])
            probe_results.append([])
            for i in range(probe_amount):
                probe_results[-1].append(model(probe)[0])
            timestamps.append(['sample_diversity', timer()])

            if min_loss < test_loss:
                count += 1 # add no-improvement count. 
                timestamps.append(['no_save', np.NaN])
            else:
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
                'results/best_model.pt')
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

pickle.dump(stats, open('results/stats.pkl', 'wb'))
pickle.dump(times, open('results/times.pkl', 'wb')) 
pickle.dump(probe_results, open('results/diversity.pkl', 'wb'))


# store all latest space variables for all sequences in dataset.
latent_results = []
for seqpath in allseqpaths:
    seq = list(SeqIO.parse(open(seqpath), 'fasta'))[0]
    seq = [int(s) for s in seq]
    seq = [seq]
    seq = torch.tensor(seq).int().to(device)
    recon_seq, s, mu = model(seq)
    #filename = seq.replace('sequences', 'results')
    
    latent_results.append([
        seqpath,
        seq.data.cpu().numpy(),
        recon_seq.data.cpu().numpy().flatten(), #possible: numpy.astype(int32)
        #''.join([str(i) for i in recon_seq.data.cpu().numpy().astype(np.int64)]),
        s.data.cpu().numpy(),
        mu.data.cpu().numpy(),
        model.z.data.cpu().numpy()])
pickle.dump(latent_results,open('results/latent_results.pkl', 'wb')) 
