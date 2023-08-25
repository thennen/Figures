from importlib import reload
import plot_config
reload(plot_config)
from plot_config import *

from matplotlib import pyplot as plt

fig, (ax1, ax2, ax3) = widefig(ncols=3)
ax1.plot(np.random.randn(100))
ax2.plot(np.random.randn(100))
ax3.plot(np.random.randn(100))

savefig('random3')