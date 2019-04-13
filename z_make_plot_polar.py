import numpy as N
import matplotlib as mpl
import matplotlib.pyplot as plt

xval = N.arange(0, 2*N.pi, 0.01)
yval = N.ones_like(xval)

colormap = plt.get_cmap('hsv')
norm = mpl.colors.Normalize(0.0, 2*N.pi)


fig = plt.figure()
# fig.set_size_inches(4, 4)


ax = plt.subplot(1, 1, 1, polar=True)
ax.scatter(xval, yval, c=xval, s=300, cmap=colormap, norm=norm, linewidths=0)
ax.set_yticks([])


import os
plt.tight_layout(pad=1, w_pad=0, h_pad=0)
plt.savefig(os.path.join('./', 'polar_plot.png'), dpi=144)
plt.close(fig)
