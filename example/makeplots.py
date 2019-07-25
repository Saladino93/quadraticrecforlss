from matplotlib import pyplot as plt

import opentext

import sys

import pickle

import numpy as np

if len(sys.argv) == 1:
        print('Choose your directory!')
        sys.exit()

direc = str(sys.argv[1])

filename = 'values.txt'
values = opentext.get_values(direc+'/'+filename)

#Specify your data directory with the N curves
data_dir = direc+'/data_dir/'

#Specify your output plot directory
output = direc+'/pics/'


rows = [values['name']]
columns = ('ngal', 'minkhrec', 'maxkhrec', 'minkhanalysis', 'maxkhanalysis', 'fnlfid', 'bgfid', 'b20', 'cg', 'cs', 'ct', 'deltac', 'a1', 'a2')

cell_text = []

for i in columns:
	cell_text.append([values[i]])

cell_text = np.array(cell_text)
cell_text.reshape((len(rows), len(columns)))

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText = cell_text.T,
                      rowLabels = rows,
                      colLabels = columns, loc = 'center')
fig.tight_layout()
plt.savefig(output+'table.png', dpi = 300)


with open(direc+'/data_dir/spectra.pickle', 'rb') as handle:
    dic = pickle.load(handle)

values = dic['values']
K = dic['K']
nbar = dic['ngal']
shot = dic['shotnoise']
PL = dic['PL']
fnl = dic['fnl']

fig, ax = plt.subplots( nrows=1, ncols=1 )
plt.title('Case for $f_{nl}='+str(fnl)+'$')
plt.xlabel('$K$ $(h Mpc^{-1})$')
plt.ylabel('$P$ $(h^{-3} Mpc^{3})$')
ax.loglog(K, PL, label = '$P_L$')
ax.loglog(K, shot+0.*K, label = 'Shot Noise')
for a in values:
	ax.loglog(K, dic['Ng'+a], label = '$N_{g'+a+'}$')
ax.legend(loc = 'best', prop = {'size': 6})
fig.savefig(output+'Noisecurves.png', dpi = 300)
plt.close(fig)


fig, ax = plt.subplots( nrows=1, ncols=1 )
plt.title('Case for $f_{nl}='+str(fnl)+'$')
plt.xlabel('$K$ $(h Mpc^{-1})$')
plt.ylabel('Contamination Abs($\kappa_{\\alpha}\\frac{N_{gg}}{N_{g\\alpha}}$)')
for a in values:
        ax.plot(K, abs(dic['k'+a]*dic['Ngg']/dic['Ng'+a]), label = '$'+a+'$')
ax.plot(K, dic['bfnllargescales'], label = '$f_{nl}$ in original field')
ax.set_yscale('log')
ax.legend(loc = 'best', prop = {'size': 6})
fig.savefig(output+'contaminationcurves.png', dpi = 300)
plt.close(fig)
