import numpy as np

def index_of_closest(query, search_space):
	'''Index of closest element to query in search_space (in L2-norm)'''
	squared_l2_dist = lambda x,y: ((x-y)**2).sum()
	res = np.argmin([squared_l2_dist(query, item) for item in search_space])
	return res

def match_node_representations(offspring, evecs):
	'''
	Match all members of offspring to the embedding space.

	Matching is necessary because after combination of genes in the embedding
	space, the result is not guaranteed to match a node existing in the
	network. Hence, the nearest node representation is picked.

	Parameters
	-----------
	offspring: np.ndarray
	offspring must have the following form:
	- first column: start time idxs
	- remaining columns: elements in the node embedding space after crossover

	evecs: np.ndarray
	eigenvectors in the embedding space
	must have the same shape as remaining columns in offspring
	
	Returns
	--------
	matched_offspring: offspring matching actual nodes in the network
	This must have the following structure:
	- first column: indices of start times
	- second column: indices of junction names
	- remaining columns: junction representations in the embedding space
	'''
	start_time_idxs = offspring[:, 0]
	unmatched_node_representations = offspring[:, 1:]
	matched_node_representations = []
	junction_name_idxs = []
	for node_rep in unmatched_node_representations:
		idx_of_closest = index_of_closest(node_rep, evecs)
		matched_node_representations.append(evecs[idx_of_closest])
		junction_name_idxs.append(idx_of_closest)
	offspring = np.column_stack((
		start_time_idxs,
		np.array(junction_name_idxs),
		np.array(matched_node_representations)
	))
	return offspring

def crossover_function_factory(evecs):
	'''
	Perform uniform crossover and match offspring afterwards.

	Matching is necessary because after combination of genes in the embedding
	space, the result is not guaranteed to match a node existing in the
	network. Hence, the nearest node representation is picked.

	Parameters
	-----------
	evecs: np.ndarray
	eigenvectors of the water network's graph Laplacian used for the embedding
	After the crossover, each member of the offspring generation is replaced
	by its nearest neighbour in the embedding space i.e. its nearest row in
	evecs

	Returns
	--------
	A crossover function that can be used by an instance of pygad.GA
	'''
	def crossover_func(parents, offspring_size, ga_instance):
		offspring = ga_instance.uniform_crossover(parents, offspring_size)
		offspring = np.delete(offspring, 1, 1) # drop junction idxs
		offspring = match_node_representations(offspring, evecs)
		return offspring
	return crossover_func

def track_mutations(mutation_tracker, old_offspring, offspring):
	'''Count number of start_time and junction mutations.'''
	mutation_tracker['start_time'] += (
		offspring[:,0]!=old_offspring[:,0]
	).sum()
	mutation_tracker['junction_name'] += (
		offspring[:,1]!=old_offspring[:,1]
	).sum()
	return

def mutation_function_factory(evecs,
		mutation_tracker=None, mutate_junctions_directly=True):
	'''
	Create a mutation function that is compliant with node embeddings.

	Parameters
	-----------
	evecs: np.ndarray
	eigenvectors of the network's graph Laplacian used for the node embedding

	mutation_tracker: dict, default=None
	if start_time and junction mutations should be tracked, you may pass a
	dictionary with keys 'start_time' and 'junction_name' and initial values
	of zero for both. Counts will be updated as the genetic algorithm
	progresses.

	mutate_junctions_directly: bool, default=True
	if true, junction ids are mutated directly. Otherwise, mutation takes
	place in the embedding space and the resulting offspring is matched to
	existing junctions. The first version appears to work better.

	Returns
	--------
	A mutation function according to the parameters. Random mutation is used.
	'''
	if mutate_junctions_directly:
		if mutation_tracker is None:
			def mutation_func(offspring, ga_instance):
				offspring = offspring[:,:2]
				offspring = ga_instance.random_mutation(offspring)
				evec_idxs = np.array(offspring[:,1], dtype=np.int64)
				node_representations = evecs[evec_idxs]
				offspring = np.column_stack([offspring, node_representations])
				return offspring
		else: # if mutation_tracker is not None
			def mutation_func(offspring, ga_instance):
				old_offspring = offspring.copy()
				offspring = offspring[:,:2]
				offspring = ga_instance.random_mutation(offspring)
				evec_idxs = np.array(offspring[:,1], dtype=np.int64)
				node_representations = evecs[evec_idxs]
				offspring = np.column_stack((offspring, node_representations))
				track_mutations(mutation_tracker, old_offspring, offspring)
				return offspring
	else: # if not mutate_junctions_directly
		if mutation_tracker is None:
			def mutation_func(offspring, ga_instance):
				offspring = np.delete(offspring, 1, 1) # drop junction idxs
				offspring = ga_instance.random_mutation(offspring)
				offspring = match_node_representations(offspring, evecs)
				return offspring
		else: # if mutation_tracker is not None
			def mutation_func(offspring, ga_instance):
				old_offspring = offspring.copy()
				offspring = np.delete(offspring, 1, 1) # drop junction idxs
				offspring = ga_instance.random_mutation(offspring)
				offspring = match_node_representations(offspring, evecs)
				track_mutations(mutation_tracker, old_offspring, offspring)
				return offspring
	return mutation_func

def random_mutation_wrapper(mutation_tracker):
	'''Return a wrapper of random mutation where mutations are tracked.'''
	def mutation_func(offspring, ga_instance):
		old_offspring = offspring.copy()
		offspring = ga_instance.random_mutation(offspring)
		mutation_tracker['start_time'] += (
			offspring[:,0]!=old_offspring[:,0]
		).sum()
		mutation_tracker['junction_name'] += (
			offspring[:,1]!=old_offspring[:,1]
		).sum()
		return offspring
	return mutation_func
