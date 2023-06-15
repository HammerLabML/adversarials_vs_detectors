import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def onclick(event):
	for bar in bars:
		event_at_bar = (
			event.ydata > bar.get_y()
			and
			event.ydata < bar.get_y() + bar.get_height()
		)
		if event_at_bar:
			bar.set_visible(not bar.get_visible())
	fig.canvas.draw()

leak_areas = pd.read_csv('max_areas.csv', index_col='junction').squeeze('columns')
#normal_junction_idxs = leak_areas.index.drop('n300')
#picked_idxs = np.random.choice(normal_junction_idxs, size=4, replace=False)
#picked_idxs = np.concatenate([picked_idxs, ['n300']])
#leak_areas = leak_areas[picked_idxs]
fig, ax = plt.subplots(figsize=(10,4.8))
y_pos = list(reversed(range(len(leak_areas))))
cmap = mpl.colormaps['RdYlGn_r']
barh_container = ax.barh(y_pos, leak_areas)
bars = barh_container.patches
normalized_leak_areas = leak_areas/250
for bar, normalized_area in zip(bars, normalized_leak_areas):
	plt.setp(bar, 'facecolor', cmap(normalized_area))
ax.set_yticks(y_pos, leak_areas.index)
ax.set_xlabel('Leak area in $cm^2$')
ax.set_ylabel('Junctions in L-Town')
fig.canvas.mpl_connect('button_press_event', onclick)
#for bar in bars:
#	bar.set_visible(False)
plt.savefig('/Users/paulstahlhofen/Documents/Water_Futures/Presentation_Cyprus_2023/Figures/sensitivity_comparison.png', dpi=300)

