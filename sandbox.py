import pickle
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

cwd = os.getcwd()
target = 'stats.pkl'
test = pickle.load(open(os.path.join(cwd, target),'rb'))

print(len(test))
plt.plot(test[0])
plt.savefig('sandbox.png')
