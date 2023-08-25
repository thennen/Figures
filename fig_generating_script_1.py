from importlib import reload
import plot_config
reload(plot_config)
from plot_config import *

import numpy as np
from matplotlib import pyplot as plt

kuwacr, kuwalogrho = np.loadtxt(pjoin(localdatadir, 'Kuwamoto.csv'), delimiter=',', skiprows=1, usecols=(1,2), unpack=True)

fig, ax1 = fig_compact()
ax1.plot(kuwacr, 10**kuwalogrho, color='C2', marker='.', label='Kuwamoto et al.\n(single crystal)')
ax1.set_yscale('log')
ax1.set_ylabel('Resistivity [Ω cm]')
ax1.set_xlabel('Cr content [%]')
ax1.legend(loc=0, framealpha=1)

savefig('kuwamoto', metastring='some extra information')

fig, (ax1, ax2) = fig1(ncols=2)
ax1.plot(np.random.randn(100))
ax2.plot(np.random.randn(100))

savefig('random2')
