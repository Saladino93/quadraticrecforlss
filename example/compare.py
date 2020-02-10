import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "stix"


import yaml

import sys

if len(sys.argv) != 3:
    print('Choose two configuration files!')
    sys.exit()

## Read configuration file

values_file_1 = str(sys.argv[1])
values_file_2 = str(sys.argv[2])

with open(values_file_1, 'r') as stream:
    data_1 = yaml.safe_load(stream)

with open(values_file_2, 'r') as stream:
    data_2 = yaml.safe_load(stream)

lista = [data_1, data_2]

title = 'Comparing '
title_fraction = 'Improvement'

xlabel = '$K$ $(h Mpc^{-1})$'
ylabel = '$\sigma_{f_nl}$'

fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True)
ax1, ax2 = ax[0], ax[1]

ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax1.set_yscale('log')

ax2.set_xlabel(xlabel)
ax2.set_title(title_fraction, size = 10)

i = 1
r = 1.
legend_frac = ''

for values in lista:
    direc = values['name']
    base_dir = values['file_config']['base_dir']
    data_dir = values['file_config']['data_dir']
    kf, err = np.loadtxt(base_dir+direc+'/'+data_dir+'sigma_fnl_marg_int.dat', unpack = True)
    ax1.plot(kf, err, label = direc, lw = 2)
        
    r *= err**i
    i -= 2
 
    legend_frac += direc+'/'
    title += direc+' vs '

title = title[:-4]
fig.suptitle(title)

legend_frac = legend_frac[:-1]
ax2.plot(kf, r, label = legend_frac, lw = 2)    
ax1.legend(loc = 'best', prop = {'size': 6})
ax2.legend(loc = 'best', prop = {'size': 6})

plt.subplots_adjust(hspace = 0.4)

fig.savefig('comparing.png', dpi = 300)
plt.close(fig)








