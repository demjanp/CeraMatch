import json
import numpy as np
from matplotlib import pyplot

fstats = "data/stats.json"

with open(fstats, "r") as f:
	stats = json.load(f)  # [[w_R, w_th, w_kap, w_h, w_diam, w_ax, n_clusters_idx, n_clusters_p, p_opt, max_clu_index], ...]; p_opt = bayesian probability of LDA assigning a profile to the right cluster
stats = np.array(stats)

def format_weights(slice):
	
	w_R, w_th, w_kap, w_h, w_diam, w_ax, n_clusters_idx, n_clusters_p, p_opt, max_clu_index = slice
	return "w_h: %0.2f, w_diam: %0.2f, w_ax: %0.2f, n_clust_idx: %d, n_clust_p: %d, p_opt: %0.2f, clu_index: %0.2f" % (w_h, w_diam, w_ax, n_clusters_idx, n_clusters_p, p_opt, max_clu_index)

to_vis = {
	"Clu. N Idx": 6,
	"Clu. N P": 7,
	"P": 8,
	"Clu. Idx": 9,
}

for title in to_vis:
	vis_idx = to_vis[title]
	slice = stats[np.argmax(stats[:,vis_idx])]
	print("\nmax %s:" % (title), format_weights(slice))
	slice = stats[np.argmin(stats[:,vis_idx])]
	print("\nmin %s:" % (title), format_weights(slice))

slice = stats[np.argmax(stats[:,3])]
print("\nmax w_h:", format_weights(slice))

slice = stats[np.argmax(stats[:,4])]
print("\nmax w_diam:", format_weights(slice))

slice = stats[np.argmax(stats[:,5])]
print("\nmax w_ax:", format_weights(slice))

w_hs, w_diams, w_axs = set(), set(), set()
for w_h, w_diam, w_ax in stats[:,3:6]:
	w_hs.add(w_h)
	w_diams.add(w_diam)
	w_axs.add(w_ax)
w_hs, w_diams, w_axs = sorted(w_hs), sorted(w_diams), sorted(w_axs)

plot_n = 0
for title in to_vis:
	vis_idx = to_vis[title]
	plot_n += 1
	
	grid = np.zeros((len(w_diams), len(w_axs)))
	for row in stats:
		w_h, w_diam, w_ax = row[3:6]
		r = w_diams.index(w_diam)
		c = w_axs.index(w_ax)
		grid[r,c] = row[vis_idx]

	cmax, rmax = np.argwhere(grid == grid.max()).T
	
	pyplot.subplot(100 + 10*len(to_vis) + plot_n)
	pyplot.title(title)
	pyplot.imshow(grid, vmin = stats[:,vis_idx].min(), vmax = stats[:,vis_idx].max())
	pyplot.plot(rmax, cmax, "+", color = "red")
	if plot_n == 1:
		pyplot.yticks(range(len(w_diams)), np.round(w_diams, 2))
	pyplot.xticks(range(len(w_axs)), np.round(w_axs, 2), rotation = "vertical")
	pyplot.xlabel("Weight Axis")
	if plot_n == 1:
		pyplot.ylabel("Weight Diameter")
	pyplot.gca().invert_yaxis()

pyplot.show()

