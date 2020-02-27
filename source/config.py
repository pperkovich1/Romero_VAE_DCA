import torch
from torch import nn
import numpy as np

prev_model = "model.pt"
msa = "metaclust_processed_msa.fasta"

input_length = 11088
num_hidden = 400
num_latent = 20
activation_func = nn.Sigmoid()
learning_rate = .001

batch_size = 5
max_epochs = 10
convergence_limit = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

AMINO_ACIDS = np.array([aa for aa in "RKDEQNHSTCYWAILMFVPG-"], "S1")
# AAs = AMINO_ACIDS[:-1] # drop the gap character
AAs = AMINO_ACIDS # idk why Sameer dropped the gap character so I'm not doing that
AAs_string = AAs.tostring().decode("ascii")
AA_L = AAs.size # alphabet size
AA_map = {a:idx for idx,a in enumerate(AAs)} # map each amino acid to an index
# same map as above but with ascii indices
AA_map_str = {a:idx for idx, a in enumerate(AAs_string)}

example_seq = '''KDAVTNMGVGWNLGNTLDANDGSKTWT-----------------TTEQHETCWGQPVTKP
ELMKMMAEAGFNTIRVPVTWYQEMDA--NGKVNDAWMKRVKEVVDYVIDNGMYCILNVHH
DTGADS-NTFK------SWLKASSKNYTANKDKYEYLWKQIAETFKDYDDHLLFEAYNEM
L-----------------DEKSTW----NEPVDKTDG-----------YKAINDYAKSFV
TTVRNTGGNNKDRNLIVNTYSASS-----------------------------------M
PNAMKNLDLPEE---S-NH-----------------------------------------
------IIFQIHSYP-----------NWQTKSN---------------------------
----------AKKEIDNLISNIKSNLLNR--APVIIGEYATFTTWPSDIDYYNTDREVAL
YAMDYLIQETKK------AGV---GTCYWMGLSDGTYRTLPVFHQADL'''
example_seq = example_seq.replace('\n','')
input_length = len(example_seq)*21
print("Input length:%d"%input_length)
