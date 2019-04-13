
import numpy as N
import argparse, os

cl = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# relu
# cl_ray_turi = [0.4223, 0.3037, 0.2552, 0.2238, 0.6089, 0.5568, 0.5075, 0.4732, 0.4217, 0.542, 0.5288, 0.5054, 0.5416, 0.4655, ]
# cl_davies_bouldin = [0.9606, 0.7503, 0.6417, 0.4555, 0.6993, 0.6658, 0.5782, 0.5509, 0.5471, 0.5788, 0.6059, 0.6148, 0.6182, 0.6104, ]

# # linear
cl_ray_turi = [0.5988, 0.4378, 0.3477, 0.2728, 0.196, 0.413, 0.3598, 0.4444, 0.42, 0.4064, 0.4097, 0.4201, 0.4463, 0.4252, ]
cl_davies_bouldin = [1.2009, 0.8067, 0.6285, 0.4857, 0.3674, 0.4575, 0.5158, 0.5521, 0.5979, 0.5918, 0.6561, 0.627, 0.6516, 0.6307, ]



fontsize = 18

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_size_inches(6, 4)
plt.plot(cl, cl_ray_turi, '-', cl, cl_davies_bouldin, '--', linewidth=3, c='black')
plt.legend(['Ray Turi', 'Davies Bouldin'], loc='best')


ax = fig.get_axes()[0]
ax.set_xlabel('Number of Clusters', fontsize=fontsize)
ax.set_xticks([2,4,5,6,8,10,12,14,16])
ax.set_xticklabels([2,4,'',6,8,10,12,14,16])

ax.set_ylabel('Cluster Index', fontsize=fontsize)
ax.set_yticks([0.2,0.3,0.5,0.7,0.9,1.2])
ax.set_yticklabels([0.2,'',0.5,0.7,0.9,1.2])

ax.grid()

for m in [ax.title] + ax.get_xticklabels() + ax.get_yticklabels():
	m.set_fontsize(fontsize)

ax.axhline(linewidth=4, color='black')
ax.axvline(linewidth=4, color='black')

plt.tight_layout(pad=1, w_pad=0, h_pad=0)
plt.savefig(os.path.join('./', 'cluster_index_plot.png'), dpi=144)
plt.close(fig)

