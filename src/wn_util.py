import wntr
import time
import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def net1():
	''' Load Net1 from a file. '''
	network_file = '../Network_Originals/Net1.inp'
	wn = wntr.network.WaterNetworkModel(network_file)
	return wn

def hanoi():
	'''Load Hanoi_CMH from a file.'''
	network_file = '../Network_Originals/Hanoi_CMH.inp'
	wn = wntr.network.WaterNetworkModel(network_file)
	return wn

def ltown_real():
	'''Load the realistic L-Town model from a file.'''
	network_file = '../Data/L-Town/L-TOWN_Real.inp'
	wn = wntr.network.WaterNetworkModel(network_file)
	return wn

def ltown_toy():
	'''Load the L-Town model with toy demands from a file.'''
	network_file = '../Data/L-Town/L-TOWN_Model.inp'
	wn = wntr.network.WaterNetworkModel(network_file)
	return wn

def print_types(wn, nodes=True, links=True):
	'''
	Print node and link types of a WaterNetworkModel.

	Parameters
	----------

	nodes: bool, default=True
	whether or not network nodes should be included in the description

	links: bool, default=True
	whether or not network links should be included in the description
	'''
	if nodes:
		node_names = wn.node_name_list
		print('Node Types')
		print('='*30)
		for node_name in node_names:
			node_type = wn.get_node(node_name).node_type
			print(f'{node_name}: {node_type}')
	if nodes and links:
		print()

	if links:
		link_names = wn.link_name_list
		print('Link Types')
		print('='*30)
		for link_name in link_names:
			link_type = wn.get_link(link_name).link_type
			print(f'{link_name}: {link_type}')

def describe_and_plot(wn, title):
	'''
	Print node and link types and plot the network graph.

	Parameters
	-----------

	wn: wntr.network.WaterNetworkModel
	the network to be printed

	title: str
	title of the network, also used for the filename

	Note:
	Currently the network graph is saved as "Net1.png".
	This is hard-coded for now. 
	'''
	print_types(wn)
	fig, ax = plt.subplots()
	wntr.graphics.plot_network(
		wn, node_labels=True, link_labels=False,
		node_alpha=0.2, node_size=10, link_width=0.5, link_alpha=0.2,
		directed = True, title=title, ax=ax
	)
	plt.savefig(f'{title}.png', dpi=300)

def measure_simulation_time(wn, n_simulations=100):
	'''
	Print the time of a given number of 1-week simulations (default: 100).
	'''
	total_time = 0
	for i in range(n_simulations):
		sim = wntr.sim.WNTRSimulator(wn)
		start = time.time()
		sim.run_sim()
		end = time.time()
		total_time += end - start
		wn.reset_initial_values()
	print(f'Total time: {total_time:.2f}')

def get_days_and_hours(time_in_seconds):
	'''Convert time in seconds to tuple (day, hour).'''
	total_hours = time_in_seconds / SECONDS_PER_HOUR
	return np.divmod(total_hours, HOURS_PER_DAY)

def place_leak(wn, leak_properties):
	'''
	Add the leak defined in leak_properties to the water network.

	Parameters
	-----------

	wn: wntr.network.WaterNetworkModel
	the model to which the leak should be added

	leak_properties: LeakProperties.LeakProperties object
	junction name, start time, area and duration of the leak (see
	documentation of LeakProperties)

	Returns
	--------
	a WaterNetworkModel containing the leak
	'''
	leak_junction = wn.get_node(leak_properties.junction_name)
	start_time = leak_properties.start_time
	leak_junction.add_leak(
		wn, area=leak_properties.area,
		start_time = start_time,
		end_time = start_time + leak_properties.duration
	)
	return wn

def compute_pressures(wn, leak_properties=None, flow_links=[]):
	'''
	Run a hydraulic simulation and return only the pressure values.

	Parameters
	-----------

	wn: wntr.network.WaterNetworkModel
	the network for which to run the simulation

	leak_properties: LeakProperties.LeakProperties object, default=None
	if this is given, a leak with these properties is placed in the network
	before running the simulation. See the documentation of the LeakProperties
	class for details

	Returns
	--------
	pressures: pd.DataFrame
	the pressures at all nodes in the network over the whole simulation time

	Note: 
	The WNTRSimulator is used for simulation rather than the
	EpanetSimulator, because the latter turned out to ignore
	the created leak.
	'''
	if leak_properties is not None:
		wn = place_leak(wn, leak_properties)
	sim = wntr.sim.WNTRSimulator(wn)
	results = sim.run_sim()
	pressures = results.node['pressure']
	if flow_links:
		flow = results.link['flowrate'][flow_links]
		flow = flow.rename(
			columns={link: f'Flow_{link}' for link in flow_links}
		)
		pressures = pd.concat((pressures, flow), axis=1)
	return pressures

def start_time_range(start, end, leak_duration, timestep):
	'''Time range for allowed leak start times.'''
	start, end, leak_duration, timestep = map(
		int, (start, end, leak_duration, timestep)
	)
	return range(start, end - leak_duration + 1, timestep)

def describe_performance(ga, best_solution, performance_path):
	'''
	Save the performance of a Genetic Algorithm to a path.

	Parameters
	-----------
	ga: pygad.GA
	genetic algorithm instance after the algorithm was run

	best_solution: LeakProperties
	best solution found by the algoritm

	performance_path: str
	path to which the result should be safed

	The result includes
	- a json-file containing information about the best solution
	- a json-file containing parameters of the algorithm
	- a plot showing the evolution of the best achieved fitness so far
	  throughout the generations.
	'''
	if performance_path[-1]!='/':
		performance_path+='/'
	if not os.path.exists(performance_path):
		os.mkdir(performance_path)
	ga_params = dict(
		num_genes=ga.num_genes,
		num_generations=ga.num_generations,
		sol_per_pop=ga.sol_per_pop,
		num_parents_mating=ga.num_parents_mating,
		parent_selection_type=ga.parent_selection_type,
		keep_parents=ga.keep_parents,
		crossover_type=str(ga.crossover_type),
		mutation_type=str(ga.mutation_type),
		mutation_probability=ga.mutation_probability
	)
	params_file = performance_path + 'params.json'
	with open(params_file, 'w') as fp:
		json.dump(ga_params, fp)
	fig = plt.figure()
	plt.plot(range(ga.num_generations + 1), ga.best_solutions_fitness)
	plt.xlabel('generation')
	plt.ylabel('best solution fitness')
	fig.suptitle('Fitness Evolution')
	fitness_evolution_file = performance_path + 'fitness_evolution.png'
	plt.savefig(fitness_evolution_file)
	best_solution_file = performance_path + 'best_solution.json'
	best_solution.to_json(best_solution_file)

def train_wn():
	'''
	An example network for training a LeakageDetector

	The network consists of 5 days of demands.
	'''
	wn = wntr.network.WaterNetworkModel('../Data/Hanoi_Example/train.inp')
	return wn

def test_wn(leak_properties=None):
	'''
	An example network for testing a LeakageDetector

	The network consists of 2 days of demands. A custom leak can be added
	(see documentation of LeakProperties).
	'''
	wn = wntr.network.WaterNetworkModel('../Data/Hanoi_Example/test.inp')
	if leak_properties is not None:
		wn = place_leak(wn, leak_properties)
	return wn

def construct_node_embedding(wn_like, junction_name_list=None):
	'''
	Map junctions of a WDN to vectors based on spectral analysis.

	This uses the 2nd to 5th eigenvector of the network's Laplacian for the
	embedding.

	Parameters
	-----------

	wn_like: either a WaterNetworkModel or a path (str) to a .inp-file
	network to use

	junction_name_list: list of str
	names of junctions for which the embedding should be computed.
	default: All junctions in the network
	This can be useful to specify a subset of junctions if some junctions are
	excluded.

	Returns
	--------
	evec_list: A list of 4 eigenvectors aligned to junction_name_list
	'''
	try:
		wn = wn_like
		G = wn.get_graph().to_undirected()
	except AttributeError:
		wn = wntr.network.WaterNetworkModel(wn_like)
		G = wn.get_graph().to_undirected()
	A = nx.to_numpy_array(G)
	D = np.diag(A.sum(axis=1))
	L = D - A
	eigenvalues, eigenvectors = np.linalg.eig(L)
	order = np.argsort(eigenvalues)
	eigenvalues = eigenvalues[order]
	eigenvectors = eigenvectors[:, order]
	eigenvectors = eigenvectors[:, 1:5]
	# keep only the values for specified junctions (if applicable)
	if junction_name_list is None:
		junction_name_list = wn.junction_name_list
	mask = [node_name in junction_name_list
			for node_name in wn.node_name_list]
	eigenvector_list = list(eigenvectors[mask].T)
	return eigenvector_list

def plot_spectral(network, evec_index=2):
	'''
	Plot a spectral clustering of a WDN.

	Parameters
	-----------
	network: either 'hanoi' or 'ltown'
	The network for which the spectrum should be plotted.

	evec_index: int
	ordinal number of the eigenvector of the graph Laplacian to use
	default: 2
	eigenvectors are sorted ascending according to their eigenvalues.

	This will show an image (no return value).
	'''
	if network=='ltown':
		network_file = '../Data/L-Town/L-TOWN_Model.inp'
	elif network=='hanoi':
		network_file = '../Network_Originals/Hanoi_CMH.inp'
	else:
		raise AttributeError(
			f"network must be either 'ltown' or 'hanoi',"
			f" {network} is not accepted."
		)
	evec_list = construct_node_embedding(network_file)
	evec = evec_list[evec_index-2]
	wn = wntr.network.WaterNetworkModel(network_file)
	junction_idxs = range(len(wn.junction_name_list))
	embedding = {wn.junction_name_list[i]: evec[i] for i in junction_idxs}
	for node_name in wn.node_name_list:
		if node_name not in wn.junction_name_list:
			embedding[node_name] = 0
	wntr.graphics.plot_network(wn, node_attribute=embedding)
	plt.show()

