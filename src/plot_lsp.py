import wn_util as wnu

import wntr
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_lsp_candidates(wn, nodes_with_sensors, lsp_candidates,
		output_file=None, figsize=None,
		extra_lsp_candidates=None, extra_lsp_candidate_style='label',
		draw_other_nodes=False):
	'''
	Visualize the least sensitve point in a WDN.

	This will show the created figure, unless output_path is set. In that
	case, the figure is saved to the given file

	Parameters
	-----------

	wn: wntr.network.WaterNetworkModel
	the network in which the LSP should be visualized

	nodes_with_sensors: list of junction names (str)
	pressure sensor locations

	lsp_candidates: list of junction names (str)
	candidates for the least sensitive point. These are labeled 'initial LSP
	candidates'.

	output_file: str, optional, default=None
	file to save the figure. If none is given, the figure is shown directly

	figsize: (float, float), optional
	width and height of the figure in inches

	extra_lsp_candidates: list of junction names (str), optional, default=None
	LSP candidates found after excluding the initial LSP candidates from the
	search space

	extra_lsp_candidate_style: one of 'label' or 'triangle', default='label'
	The way how the extra lsp candidates are shown. 'label' means they are
	labeled with the junction name

	draw_other_nodes: bool, optional, default=False
	If true, other network nodes are also drawn

	This method has no return value.
	'''
	G = wn.get_graph().to_undirected()
	pos = wn.query_node_attribute('coordinates')
	fig, ax = plt.subplots(figsize=figsize)
	nx.draw_networkx_nodes(
		nx.subgraph(G, lsp_candidates), pos,
		node_size=20, node_shape='v', node_color='red',
		label='Initial LSP candidates'
	)
	res_and_tanks = wn.reservoir_name_list + wn.tank_name_list
	nx.draw_networkx_nodes(
		nx.subgraph(G, res_and_tanks), pos,
		node_size=20, node_shape='s', node_color='blue',
		label='reservoirs and tanks'
	)
	nx.draw_networkx_nodes(
		nx.subgraph(G, nodes_with_sensors), pos,
		node_size=20, node_shape='d', node_color='green',
		label='pressure sensors'
	)
	if extra_lsp_candidates:
		if extra_lsp_candidate_style=='label':
			nx.draw_networkx_labels(G.subgraph(extra_lsp_candidates), pos)
		elif extra_lsp_candidate_style=='triangle':
			nx.draw_networkx_nodes(
				G.subgraph(extra_lsp_candidates), pos,
				node_size=20, node_shape='^', node_color='orange',
				label='LSP in reduced search space'
			)
		else:
			allowed_styles = ['label', 'triangle']
			raise ValueError(
				f"'extra_lsp_candidate_style' must be one of {allowed_styles}"
			)
	if draw_other_nodes:
		other_nodes = [
			node for node in wn.junction_name_list
			if node not in nodes_with_sensors and node not in lsp_candidates
			and node not in extra_lsp_candidates
		]
		nx.draw_networkx_nodes(
			nx.subgraph(G, other_nodes), pos,
			node_size=15, alpha=0.2, label='other nodes'
		)
	nx.draw_networkx_edges(G, pos, alpha=0.4)
	ax.axis('off')
	plt.tight_layout()
	plt.legend()
	if output_file is not None:
		plt.savefig(output_file, dpi=1000)
	else:
		plt.show()

def plot_lsp_hanoi():
	'''Plot the LSP in Hanoi based on experimental results.'''
	wn = wnu.hanoi()
	nodes_with_sensors = ['4', '13', '16', '22', '31']
	lsp_candidates = ['2', '3', '20']
	output_file = '../Writing/Figures/lsp_candidates_hanoi.eps'
	extra_lsp_candidates = ['10', '23']
	plot_lsp_candidates(
		wn, nodes_with_sensors, lsp_candidates,
		extra_lsp_candidates=extra_lsp_candidates,
		output_file=output_file
	)

def plot_lsp_ltown():
	'''Plot the LSP in L-Town based on experimental results.'''
	wn = wnu.ltown_toy()
	sensor_path = '../Data/L-Town/pressure_sensors.txt'
	nodes_with_sensors = np.loadtxt(sensor_path, dtype=str)
	lsp_candidates = [f'n{i}' for i in [44, 111, 300, 303, 336, 343]]
	output_file = '../Writing/Figures/lsp_candidates_ltown.eps'
	extra_lsp_candidates = ['n387']
	extra_lsp_candidate_style = 'triangle'
	plot_lsp_candidates(
		wn, nodes_with_sensors, lsp_candidates,
		draw_other_nodes=False, extra_lsp_candidates=extra_lsp_candidates,
		extra_lsp_candidate_style=extra_lsp_candidate_style,
		output_file=output_file
	)

if __name__=='__main__':
	plot_lsp_ltown()

