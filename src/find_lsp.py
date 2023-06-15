import wntr
from sklearn.metrics import f1_score
import pygad
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from os import getpid
from time import sleep

from bisection import bisection_search
from time_constants import SECONDS_PER_HOUR, HOURS_PER_DAY, SECONDS_PER_DAY
import wn_util
from LeakageDetectors import SingleSensorForecaster, BetweenSensorInterpolator
from LeakProperties import LeakProperties
import ga_func

# I know, this is a bit ugly, but the UserWarnings caused by the WNTRSimulator
# are not analyzed by this script and I could not turn them off locally.
import warnings
warnings.filterwarnings('ignore')

class LspFinder():
	'''
	Tool to find the least sensitve point in a water network.

	The least sensitive point is the junction at which one could place the
	largest possible leak that still remains undetected.

	Parameters
	-----------

	network_file: str, name of an EPANET inp-file
	file containing the network to be analyzed

	leak_duration: int
	duration of each tested leak in seconds

	leakage_detector: an instance of a subclass of AbstractLeakageDetector
	the leakage detector must have already been trained (see documentation of
	leakage_detactor.train) with suitable pressure values. These should be
	produced by a leak-free simulation of the same network.

	start_time_range: list or range object
	potential starting times for a leak. Note that a leak starting at the
	very beginning of the simulation might not be detected by a leakage
	detector using previous timesteps.  Also, the leak should be fixed
	before the end of the time series.  Otherwise one could compare leaks
	with different duration.

	sim_start: pd.Timestamp, optional, default: pd.Timestamp(0)
	timestamp for the beginning of the simulation (e.g. 1997-04-06 19:04:00)
	This is useful to convert all time offsets in seconds given by wntr into
	timestamps which reflect useful information like the hour of a day or
	summer/winter periods.
	Note: In the pandas implementation, Timestamp(0) evaluates to
	'1970-01-01 00:00'

	network_preparation_method: callable, default=None
	If given, this function will be called on the network after loading it
	from the file and before running a simulation. This can e.g. be used for
	setting options that would require complicated changes in the inp-file.

	ignore_nodes: list of str, optional, default: empty list
	list of junction names that should be ignored in the search for the least
	sensitive point. This can be useful to exclude "boring" junctions.

	search_info: dict, default: {'search_steps': {}}
	this stores information about search parameters and results and can be
	written to a YAML file using self.write_search_info. The user is free to
	add and fill fields for logging purposes
	'''

	def __init__(self, network_file, leak_duration, leakage_detector, start_time_range, sim_start=pd.Timestamp(0), network_preparation_method=None, ignore_nodes=[], search_info=dict(search_steps=dict())):
		self.id = getpid()
		self.network_file = network_file
		self.leak_duration = leak_duration
		self.leakage_detector = leakage_detector
		self.start_time_range = start_time_range
		self.sim_start = sim_start
		self.network_preparation_method = network_preparation_method
		
		wn = wntr.network.WaterNetworkModel(network_file)
		self.junction_name_list = wn.junction_name_list
		for node_name in ignore_nodes:
			self.junction_name_list.remove(node_name)

		self.bisection_search_cache = dict(
			start_times=list(self.start_time_range).copy(),
			junctions=self.junction_name_list.copy()
		)
		self.genetic_search_cache = dict()
		self.search_info = search_info
		self.search_info['id'] = self.id
		self.search_info['excluded_nodes'] = ignore_nodes

	def info(self, msg):
		'''Useful for logging'''
		print(f'(LspFinder {self.id}) {msg}')

	def remove_node(self, node_name):
		'''Remove a node from junction name list and search caches.'''
		self.junction_name_list.remove(node_name)
		self.bisection_search_cache['junctions'].remove(node_name)
		cache_set = bool(self.genetic_search_cache)
		if (cache_set and
				self.genetic_search_cache['junction_name']==node_name):
			self.genetic_search_cache = dict()

	def absolute_time(self, offset):
		'''Convert an offset in seconds to a pd.Timestamp.'''
		return self.sim_start + pd.Timedelta(seconds=offset)

	def place_leak_and_detect(self, leak_properties):
		'''
		Detect a leak with given properties.

		self.network_file is used to load the network in which the leak is
		placed.

		Parameters
		-----------

		leak_properties: LeakProperties.LeakProperties object
		this is passed to compute_pressures. See documentation of
		the LeakProperties class for details

		Returns
		--------
		an array containing the times in seconds since the simulation start
		at which an alarm was raised. An empty array corresponds to no alarm
		'''
		for i in range(10):
			try:
				wn = wntr.network.WaterNetworkModel(self.network_file)
				break
			except RuntimeError:
				self.info(
					f'Problems with file system.'
					f' Trying again in 5 minutes'
				)
				sleep(300)
		if self.network_preparation_method is not None:
			wn = self.network_preparation_method(wn)
		pressures = wn_util.compute_pressures(
			wn, leak_properties=leak_properties
		)
		alarm_times = self.leakage_detector.detect(pressures)
		return alarm_times

	def alarms_for_area(self, leak_area, *, start_time,
			use_cache=False, store_history=True):
		'''
		Place a leak of equal area at each junction in the network.

		This re-constructs the network and runs a seperate simulation
		for each junction.

		Parameters
		-----------

		leak_area: float
		area in m^2, passed to self.place_leak_and_detect

		start_time: int, required key-word argument
		start time of the leak in seconds

		use_cache: bool, default=False
		If true, self.bisection_search_cache['junctions'] will be used to
		store the junctions where a leak has not been detected after every
		run. In the next run, only these junctions will be analyzed.

		Note: The calling function is resoponsible for restoring the cache
		afterwards.

		store_history: bool, default=True
		If true, the history of function results will be stored in
		self.search_info['search_steps'] in form of a dictionary
		Keys: leak_area value
		Values: list of names of junctions where the given area did NOT lead
		to an alarm

		Returns
		--------
		num_alarms: int
		The numer of junctions for which an alarm was triggered during leak
		time. If use_cache=True, all junctions outside the cache are assumed
		to produce an alarm as well.

		lsp_candidates: list of str
		The names of all the junctions for which no alarm was triggered.
		These remain candidates for the least sensitive point.
		'''
		self.info(f'Leak area: {leak_area * 10000:.2f} cm^2')
		num_alarms = 0
		lsp_candidates = []
		if use_cache:
			junction_name_list = self.bisection_search_cache['junctions']
		else:
			junction_name_list = self.junction_name_list
		for junction_name in junction_name_list:
			self.info(f'Placing leak at node {junction_name}')
			leak_properties = LeakProperties(
				area=leak_area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			alarm_times = self.place_leak_and_detect(leak_properties)
			end_time = start_time + leak_properties.duration
			alarm_at_leak_time = np.logical_and(
				(alarm_times >= start_time),
				(alarm_times < end_time)
			).any()
			if not alarm_at_leak_time:
				self.info(
					f'Leak with size {leak_area * 10000:.2f} cm^2'
					f' at node {junction_name}'
					f' triggered no alarm.'
				)
				lsp_candidates.append(junction_name)
		num_alarms = len(self.junction_name_list) - len(lsp_candidates)
		if use_cache and lsp_candidates: # list is not empty
			self.bisection_search_cache['junctions'] = lsp_candidates
		if store_history:
			self.search_info['search_steps'][f'{leak_area * 10000:.2f}'] = lsp_candidates
		self.info('*'*30 + '\n')
		return num_alarms, lsp_candidates

	def maximize_leak_area(self, initial_area, junction_name, start_time, maximization_trials, lower_bound=None, upper_bound=None, verbose=False):
		'''
		Maximize the leak area for a junction, s.t. no alarm is triggered.

		Parameters
		-----------

		initial_area: float
		an initial guess for the leakage area in cm^2

		junction_name: str
		name of a network junction where the leak should be placed

		start_time: int, start time of the leak in seconds

		maximization_trials: int, must be positive
		number of trials in the leak maximization process

		lower_bound: float, default=None
		a leak-area value in m^2 which is known to produce NO alarm

		upper_bound: float, default=None
		a leak-area value in m^2 which is known to produce an alarm

		verbose: bool, default=False
		If True, the initial leak area and the results of the maximization
		trials will be printed

		Returns
		--------
		max_area: float
		the maximal leak area which does not produce an alarm
		'''
		if verbose:
			self.info(f'Maximize the leak area for junction {junction_name}')
		# Take some area and check if it produces a leak, for a fixed junction
		def check_leak_at_junction(area):
			leak_properties = LeakProperties(
				area=area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			alarm_times = self.place_leak_and_detect(leak_properties)
			end_time = start_time + leak_properties.duration
			alarm_at_leak_time = np.logical_and(
				(alarm_times >= start_time),
				(alarm_times < end_time)
			).any()
			alarm_string = 'alarm' if alarm_at_leak_time else 'no alarm'
			if verbose:
				self.info(f'Leak area: {area * 10000:.2f} cm^2 -> {alarm_string}')
			return int(alarm_at_leak_time)
		# Running bisection_search with a sought_result of 0.5 will never
		# finish before the end of all trials, but get closer to the point
		# where the area starts to trigger the alarm. The highest area below
		# this threshold is the maximal area we are looking for.
		_, bounds = bisection_search(
			check_leak_at_junction, 0.5,
			initial_area, maximization_trials,
			step_parameters = {'downward_scale': 2, 'upward_scale': 1.2},
			lower_bound=lower_bound, upper_bound=upper_bound
		)
		max_area = bounds[0] if bounds[0] is not None else 0
		return max_area

	def test_leak_over_time(self, area):
		'''
		Test a leak with fixed area across all nodes and timesteps.

		For each timestep, this method will record the junctions where a leak
		with size 'leak_area' did NOT trigger an alarm. It will use
		self.bisection_search_cache both to read relevant junctions and
		timesteps in the beginning of the search and to store junctions and
		timesteps that remain relevant for further search after one
		iteration. Hence, if you call this method again with a different
		area, it will presumably perform much faster.
		
		Note: The calling function is resoponsible for restoring the cache
		afterwards.

		Parameters
		-----------

		area: float
		leak area in m^2

		Returns
		--------
		n_excluded: int
		number of junctions, which are definitely NOT the
		least sensitive point. These are junctions, at which a leak
		of the given area always triggered an alarm.

		unnoticed: dictionary
		the keys are start times of the leaks in seconds, the values (str)
		are names of the network junctions.  For each time, the names of the
		junctions for which NO alarm was triggered are given.
		'''
		# Remember start times which remain candidates
		self.info(f'Trying area: {area * 10000:.2f} cm^2')
		start_time_candidates = []
		junction_candidates = set()
		unnoticed = dict()
		for start_time in self.bisection_search_cache['start_times']:
			unnoticed[start_time] = []
			self.info(
				f'Placing leak starting at {self.absolute_time(start_time)}'
			)
			for junction_name in self.bisection_search_cache['junctions']:
				leak_properties = LeakProperties(
					area=area,
					junction_name=junction_name,
					start_time=start_time,
					duration=self.leak_duration
				)
				alarm_times = self.place_leak_and_detect(leak_properties)
				end_time = start_time + leak_properties.duration
				alarm_at_leak_time = np.logical_and(
					(alarm_times >= start_time),
					(alarm_times < end_time)
				).any()
				if not alarm_at_leak_time:
					junction_candidates.add(junction_name)
					unnoticed[start_time].append(junction_name)
					self.info(
						f'-- Leak at node {junction_name}'
						f' triggered no alarm!'
					)
			if unnoticed[start_time]: # list is not empty
				start_time_candidates.append(start_time)
		self.info('*'*30 + '\n')
		if junction_candidates:
			self.bisection_search_cache['start_times'] = start_time_candidates
			self.bisection_search_cache['junctions'] = junction_candidates
		# This is imported to make the function work with bisection_search
		n_excluded = len(self.junction_name_list) - len(junction_candidates)
		return n_excluded, unnoticed

	def find_lsp_bisection(self, *, start_time_trials, maximization_trials, trials_per_timestep, global_trials, initial_area=0.01):
		'''
		Use bisection_search to determine the least sensitive point.

		The procedure of this function consists of two main steps:
		1. for a fixed number of time trials...
			1.1 find the least sensitive point at that time
			1.2 maximize the leak area for that time at the least sensitive
			point such that the leakage detection algorithm causes no alarm.
		Subsequently, use the highest of the maxima determined above
		as a starting point for step 2
		2. find the least sensitive point globally (over all starting times)

		In each of the substeps, binary search is applied to find a leakage
		area fulfilling the condition. In case of 1.1 and 2, the area is
		picked such that it does NOT trigger an alarm in one node, while it
		does trigger an alarm in all other nodes. See bisection_search for
		the search procedure.  For the global lest-sensitive-point search
		(Step 2), see self.test_leak_over_time.

		Parameters
		-----------
	
		start_time_trials: int, required key-word argument
		number of different leak starting times in phase 1

		trials_per_timestep: int, required key-word argument
		number of search trials per starting time in phase 1

		maximization_trials: int, required key-word argument
		number of trials to maximize the leak area in phase 1.2 (passed to
		maximize_leak_area)

		global_trials: int, required key-word argument
		maximum amount of search_leak_over_time calls in phase 2

		initial_area: float, area in m^2, default=0.01
		the area that is used at the beginning of the search in each time
		trial in step 1.1

		Returns
		--------
		the name of the junction where the largest undetected leak could be
		placed
		'''
		start_times = np.random.choice(
			self.start_time_range,
			size=start_time_trials,
			replace=False
		)
		max_area = 0
		for index, start_time in enumerate(start_times):
			self.info('='*15 + f'TIME TRIAL {index + 1}' + '='*15)
			end_time = start_time + self.leak_duration
			print(
				f'Time parameters of the leak\n'
				f'Start: {self.absolute_time(start_time)}\n'
				f'End: {self.absolute_time(end_time)}\n'
			)
			leak_area, bounds, lsp_candidates = bisection_search(
				self.alarms_for_area,
				len(self.junction_name_list) - 1,
				initial_area,
				trials_per_timestep,
				meta_info=True,
				start_time=start_time,
				store_history=False
			)
			lsp = lsp_candidates[0]
			self.info(f'The least sensitive point is node {lsp}\n')
			new_max_area = self.maximize_leak_area(
				leak_area, lsp, start_time,
				maximization_trials,
				lower_bound=leak_area,
				upper_bound=bounds[1],
				verbose=True
			)
			self.info(
				f'The maximum leak area at the least sensitive point'
				f' (node {lsp}) and the given time parameters (see above)'
				f' is {new_max_area * 10000:.2f} cm^2.\n'
			)
			max_area = max([new_max_area, max_area])
		print()
		self.info(
			f'Maximum leak area across the time trials:'
			f' {max_area * 10000:.2f} cm^2'
		)
		print('#'*30 + '\n')
		leak_area, bounds, unnoticed = bisection_search(
			self.test_leak_over_time, 
			len(self.junction_name_list) - 1,
			max_area,
			global_trials,
			lower_bound=max_area,
			meta_info=True
		)
		total_lsp = self.bisection_search_cache['junctions'].pop()
		print(
			f'The total least sensitive point across all starting times'
			f' is junction {total_lsp}.'
		)
		# restore bisection search cache
		self.bisection_search_cache = dict(
			start_times=list(self.start_time_range).copy(),
			junctions=self.junction_name_list.copy()
		)
		return total_lsp

	def fitness_function_factory(self, lookup_table,
			maximization_trials=10, initial_area=0.01):
		'''
		Produce the fitness function for the 'genetic' and 'spectral'
		approaches.

		See documentation of 'find_lsp_genetic' and 'find_lsp_spectral'.

		This should receive and empty dictionary 'lookup' as input to
		parametrize the fitness function.
		'''
		# Code from https://stackoverflow.com/questions/69544556/passing-arguments-to-pygad-fitness-function
		def fitness_function(solution, solution_idx):
			self.info(f'Solution: {solution_idx}')
			start_time_idx, junction_name_idx = solution[0], solution[1]
			if (start_time_idx, junction_name_idx) in lookup_table.keys():
				return lookup_table[(start_time_idx, junction_name_idx)]
			start_time = self.start_time_range[start_time_idx]
			junction_name = self.junction_name_list[junction_name_idx]
			# self.genetic_search_cache will be set
			# after the first evaluation of this fitness function
			if self.genetic_search_cache:
				# Check if the current combination can tolerate
				# a greater leak
				greatest_leak_area = self.genetic_search_cache['area']
				leak_properties = LeakProperties(
					area=greatest_leak_area,
					junction_name=junction_name,
					start_time=start_time,
					duration=self.leak_duration
				)
				alarm_times = self.place_leak_and_detect(leak_properties)
				end_time = start_time + leak_properties.duration
				alarm_at_leak_time = np.logical_and(
					(alarm_times >= start_time),
					(alarm_times < end_time)
				).any()
				if alarm_at_leak_time:
					lookup_table[(start_time_idx, junction_name_idx)] = 0
					return 0
				else:
					# if the greatest leak area so far did not create an alarm
					# try to increase the area
					leak_area = self.maximize_leak_area(
						greatest_leak_area, junction_name, start_time,
						maximization_trials, verbose=True
					)
			else: # in the very first function evaluation
				leak_area = self.maximize_leak_area(
					initial_area, junction_name, start_time,
					maximization_trials, verbose=True
				)
			self.genetic_search_cache['start_time'] = start_time
			self.genetic_search_cache['junction_name'] = junction_name
			self.genetic_search_cache['area'] = leak_area
			lookup_table[(start_time_idx, junction_name_idx)] = leak_area
			return leak_area
		return fitness_function

	def find_lsp_spectral(self, maximization_trials=10, initial_area=0.01,
			performance_path=None, reset_search_cache=True,
			save_last_generation=True, load_last_generation=False,
			track_mutations=False, verbose=False):
		'''
		Find the least sensitive point with a spectral node embedding.

		Similar to find_lsp_genetic, but junction names are represented by a
		spectral embedding. The 2nd to 5th eigenvector of the graph Laplacian
		is used to construct the embedding. The embedding itself consists of
		a 4-dimensional vector for each junction, which contains the
		corresponding elements of the eigenvectors for this junction. The
		gene space looks like this: [start_time_idxs, junction_name_idxs,
		evec2, evec3, evec4, evec5] For crossover, the junciton names are
		ignored. The crossover offspring is mapped to its nearest neighbour
		in the embedding space that actually exists in the network. For
		mutation, the embedding space is ignored and only the start_time_idxs
		or junction_name_idxs can be mutated.  The embedding representation
		is adjusted to the junction_name_idx afterwards.

		Parameters
		-----------

		maximization_trials: int, optional, default=10
		number of bisection search trials that are used to maximize the leak
		area. A higher number of trials will produce more accurate results at
		the cost of computation time.
		
		initial_area: float, optional, default=0.01
		The leak area that is used as a starting point for the maximization
		of the first solution candidate. For all following evaluations of
		the fitness function, the greatest leak area found so far is used as
		the starting point.

		performance_path: str, optional, default=None
		If the name of a directory is given here, information about the
		performance of the algorithm will be written to that directory. In
		case the directory does not exist yet, a new one will be created at
		the given path. The performance information include leak properties
		of the best attack found, parameter settings of the genetic
		algorithm and the evolution of the best achieved fitness value over
		the generations.  See wn_util.describe_performance for details.

		reset_search_cache: bool, default=True
		If True, self.genetic_search_cache is reset before the next call of
		this function. This is useful if one wants to run multiple trials
		starting from zero. If this is set to false, a maximum leak area of
		0 might be returned. In this case, the least sensitive point from
		one of the previous runs could not be improved.

		save_last_generation: bool, optional, default=True
		If True, save the last generation created by the genetic algorithm to
		a file called 'last_generation.npy' in the Resources folder

		load_last_generation: bool, optional, default=False
		If True, load a population from the file 'last_generation.npy' in the
		Resources folder which must have been created by previous runs of the
		algorihtm and use it as initial population.

		track_mutations: bool, default=False
		if True, the number of junction and start time mutations will be
		printed in the end. This was used for fault diagnosis.

		verbose: bool, optional, default=False
		In addition to the usual output, print the ordinal number of the
		generation in which the least sensitive point was found. This can be
		used for fault diagnosis.

		Returns
		--------
		the junction name of the least sensitive point
		'''
		lookup_table = dict()
		fitness_function = self.fitness_function_factory(
			lookup_table,
			maximization_trials=maximization_trials,
			initial_area=initial_area
		)
		start_time_idxs = np.arange(
			len(self.start_time_range), dtype=np.int64
		)
		junction_name_idxs = np.arange(
			len(self.junction_name_list), dtype=np.int64
		)
		gene_space = [start_time_idxs, junction_name_idxs]
		evec_list = wn_util.construct_node_embedding(
			self.network_file, self.junction_name_list
		)
		gene_space.extend(evec_list)
		evecs = np.array(evec_list).T # needed in this form later
		gene_type = [np.int64] * 2 + [np.float64] * 4
		num_genes = 6

		num_generations = 50
		n_junctions = len(self.junction_name_list)
		sol_per_pop = 20 if n_junctions >= 20 else n_junctions
		if load_last_generation:
			initial_population = np.load(
				'Resources/last_generation.npy', allow_pickle=True
			)
		else:
			initial_start_time_idxs = np.random.choice(
				start_time_idxs, size=sol_per_pop, replace=False
			)
			initial_junction_name_idxs = np.random.choice(
				junction_name_idxs, size=sol_per_pop, replace=False
			)
			initial_node_representations = evecs[initial_junction_name_idxs]
			initial_population = np.column_stack((
				initial_start_time_idxs,
				initial_junction_name_idxs,
				initial_node_representations
			))
		reproduction_rate = 0.25
		npm = int(sol_per_pop * reproduction_rate)
		num_parents_mating = npm if npm > 2 else 2
		parent_selection_type = 'sss'
		keep_parents = 1
		crossover_type = ga_func.crossover_function_factory(evecs)
		if track_mutations:
			mutation_tracker = dict(start_time=0, junction_name=0)
			mutation_type = ga_func.mutation_function_factory(
				evecs, mutation_tracker
			)
		else:
			mutation_type = ga_func.mutation_function_factory(evecs)
		mutation_probability = 0.1
		save_solutions = False

		ga_instance = pygad.GA(
			fitness_func=fitness_function,
			gene_space=gene_space,
			gene_type=gene_type,
			num_genes=num_genes,
			num_generations=num_generations,
			sol_per_pop=sol_per_pop,
			initial_population=initial_population,
			num_parents_mating=num_parents_mating,
			parent_selection_type=parent_selection_type,
			keep_parents=keep_parents,
			crossover_type=crossover_type,
			mutation_type=mutation_type,
			mutation_probability=mutation_probability,
			save_solutions=save_solutions
		)
		ga_instance.run()
		ga_solution, leak_area, _ = ga_instance.best_solution()
		start_time_idx, junction_name_idx = ga_solution[0], ga_solution[1]
		start_time = self.start_time_range[start_time_idx]
		junction_name = self.junction_name_list[junction_name_idx]
		if performance_path is not None:
			lsp_properties = LeakProperties(
				area=leak_area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			wn_util.describe_performance(
				ga_instance, lsp_properties, performance_path
			)
		self.info(
			f'The least sensitive point is junction {junction_name}'
			f' at {self.absolute_time(start_time)}'
			f' with a leak area of {leak_area*10000:.2f} cm^2'
		)
		if track_mutations:
			self.info(mutation_tracker)
		if verbose:
			bsg = ga_instance.best_solution_generation
			self.info(f'The best solution was found in generation {bsg}.')
		if reset_search_cache:
			self.genetic_search_cache = dict()
		if save_last_generation:
			np.save('Resources/last_generation.npy', ga_instance.population)
		return junction_name


	def find_lsp_genetic(self, maximization_trials=10, initial_area=0.01,
			performance_path=None, reset_search_cache=True,
			save_last_generation=True, load_last_generation=False,
			track_mutations=False, verbose=False):
		'''
		Find the least sensitive point with a genetic algorithm and bisection.

		This will start a genetic algorithm with the two different genes
		'start_time' and 'network_junction'. To evaluate the fitness
		function, bisection search is used to find the maximal leak area at
		the given junction and point in time, for which no alarm is
		triggered (see self.maximize_leak_area). To avoid running the
		maximization simulations for each candidate, the greatest leak area
		found so far is stored in self.genetic_search_cache['area']. Before
		the maximization, a leak of that area is placed at the given
		junction-time combination: If it triggeres an alarm, the candidate
		does not yield an improvement to the highest fitness value found so
		far. Hence, the fitness function will return 0 in these cases.

		Parameters
		-----------

		maximization_trials: int, optional, default=10
		number of bisection search trials that are used to maximize the leak
		area. A higher number of trials will produce more accurate results at
		the cost of computation time.
		
		initial_area: float, optional, default=0.01
		The leak area that is used as a starting point for the maximization
		of the first solution candidate. For all following evaluations of
		the fitness function, the greatest leak area found so far is used as
		the starting point.

		performance_path: str, optional, default=None
		If the name of a directory is given here, information about the
		performance of the algorithm will be written to that directory. In
		case the directory does not exist yet, a new one will be created at
		the given path. The performance information include leak properties
		of the best attack found, parameter settings of the genetic
		algorithm and the evolution of the best achieved fitness value over
		the generations.  See wn_util.describe_performance for details.

		reset_search_cache: bool, default=True
		If True, self.genetic_search_cache is reset before the next call of
		this function. This is useful if one wants to run multiple trials
		starting from zero. If this is set to false, a maximum leak area of
		0 might be returned. In this case, the least sensitive point from
		one of the previous runs could not be improved.

		save_last_generation: bool, optional, default=True
		If True, save the last generation created by the genetic algorithm to
		a file called 'last_generation.npy' in the Resources folder

		load_last_generation: bool, optional, default=False
		If True, load a population from the file 'last_generation.npy' in the
		Resources folder which must have been created by previous runs of the
		algorihtm and use it as initial population.

		track_mutations: bool, default=False
		if True, the number of junction and start time mutations will be
		printed in the end. This was used for fault diagnosis.

		verbose: bool, optional, default=False
		In addition to the usual output, print the ordinal number of the
		generation in which the least sensitive point was found. This can be
		used for fault diagnosis.

		Returns
		--------
		the junction name of the least sensitive point
		'''
		lookup_table = dict()
		fitness_function = self.fitness_function_factory(
			lookup_table,
			maximization_trials=maximization_trials,
			initial_area=initial_area
		)
		start_time_idxs = np.arange(
			len(self.start_time_range), dtype=np.int64
		)
		junction_name_idxs = np.arange(
			len(self.junction_name_list), dtype=np.int64
		)
		gene_space = [start_time_idxs, junction_name_idxs]
		gene_type = [np.int64, np.int64]
		num_genes = 2

		num_generations = 50
		n_junctions = len(self.junction_name_list)
		sol_per_pop = 20 if n_junctions >= 20 else n_junctions
		if load_last_generation:
			initial_population = np.load(
				'Resources/last_generation.npy', allow_pickle=True
			)
		else:
			initial_start_time_idxs = np.random.choice(
				start_time_idxs, size=sol_per_pop, replace=False
			)
			initial_junction_name_idxs = np.random.choice(
				junction_name_idxs, size=sol_per_pop, replace=False
			)
			initial_population = np.column_stack(
				(initial_start_time_idxs, initial_junction_name_idxs)
			)
		reproduction_rate = 0.25
		npm = int(sol_per_pop * reproduction_rate)
		num_parents_mating = npm if npm > 2 else 2
		parent_selection_type = 'sss'
		keep_parents = 1
		crossover_type = 'uniform'
		if track_mutations:
			mutation_tracker = dict(start_time=0, junction_name=0)
			mutation_type = ga_func.random_mutation_wrapper(mutation_tracker)
		else:
			mutation_type = 'random'
		mutation_probability = 0.1
		save_solutions = False

		ga_instance = pygad.GA(
			fitness_func=fitness_function,
			gene_space=gene_space,
			gene_type=gene_type,
			num_genes=num_genes,
			num_generations=num_generations,
			sol_per_pop=sol_per_pop,
			initial_population=initial_population,
			num_parents_mating=num_parents_mating,
			parent_selection_type=parent_selection_type,
			keep_parents=keep_parents,
			crossover_type=crossover_type,
			mutation_type=mutation_type,
			mutation_probability=mutation_probability,
			save_solutions=save_solutions
		)
		ga_instance.run()
		ga_solution, leak_area, _ = ga_instance.best_solution()
		start_time_idx, junction_name_idx = ga_solution
		start_time = self.start_time_range[start_time_idx]
		junction_name = self.junction_name_list[junction_name_idx]
		if performance_path is not None:
			lsp_properties = LeakProperties(
				area=leak_area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			wn_util.describe_performance(
				ga_instance, lsp_properties, performance_path
			)
		self.info(
			f'The least sensitive point is junction {junction_name}'
			f' at {self.absolute_time(start_time)}'
			f' with a leak area of {leak_area*10000:.2f} cm^2'
		)
		if track_mutations:
			self.info(mutation_tracker)
		if verbose:
			bsg = ga_instance.best_solution_generation
			self.info(f'The best solution was found in generation {bsg}.')
		if reset_search_cache:
			self.genetic_search_cache = dict()
		if save_last_generation:
			np.save('Resources/last_generation.npy', ga_instance.population)
		return junction_name

	def find_lsp(self, algorithm, **params):
		'''
		Find the least sensitive point using the given algorithm.

		Currently, the following approaches are implemented:
		'bisection': See find_lsp_bisection
		'genetic': See find_lsp_genetic
		'spectral': See find_lsp_spectral

		Please note that the parameters (**params) are different for
		the algorithms

		Parameters
		-----------

		algorithm: str, one of 'bisection', 'genetic' or 'spectral'
		search algorithm to use

		**params:
		parameters passed to the algorithms as key-word arguments. These
		differ between algorithms (see documentation of the different
		find_lsp functions)

		Returns
		--------
		The junction name of the least sensitive point
		'''
		allowed_algorithms = ['bisection', 'genetic', 'spectral']
		if algorithm not in allowed_algorithms:
			raise ValueError(f'algorithm must be one of {allowed_algorithms}')
		method_name = f'find_lsp_{algorithm}'
		return self.__getattribute__(method_name).__call__(**params)

	def write_search_info(self, output_file):
		'''Write self.search_info to output_file (str) in YAML-format.'''
		for i in range(10):
			try:
				with open(output_file, 'w') as fp:
					yaml.dump(
						self.search_info, fp,
						sort_keys=False, default_flow_style=False
					)
				break
			except RuntimeError:
				self.info(
					f'Problems with file system.'
					f' Trying again in 5 minutes'
				)
				sleep(300)



	def fixed_time_bisection(self, start_time,
			initial_area=0.01, output_file=None):
		'''
		Determine the least sensitive point at a fixed timestep.

		In order to find the lsp, bisection is used together with junction
		removal: All junctions at which an alarm was triggered are removed in
		the next step, if there was at least one junction where the same leak
		area caused no alarm. A maximum of 10 maximization trials will be
		used. In the end, the list of remaining lsp candidates is printed.

		Parameters
		-----------

		start_time: int, time in seconds
		the starting time of the leak.
		
		initial_area: float, default=0.01
		initial area of the leak in m^2

		output_file: str, optional
		if this is given, self.search_info will be written to output_file
		in YAML format.
		'''
		leak_area, bounds, lsp_candidates = bisection_search(
			self.alarms_for_area,
			len(self.junction_name_list) - 1,
			initial_area,
			10,
			meta_info=True,
			start_time=start_time,
			use_cache=True
		)
		self.info(f'leak area: {leak_area}')
		self.info(f'lsp candidates: {lsp_candidates}')
		# restore bisection search cache
		self.bisection_search_cache = dict(
			start_times=list(self.start_time_range).copy(),
			junctions=self.junction_name_list.copy()
		)
		if output_file is not None:
			self.write_search_info(output_file)

	@classmethod
	def ltown(cls, train_demands='real', dont_train=False, **kwargs):
		'''Create an LspFinder with default settings for L-Town.'''
		def network_preparation_method(wn):
			wn.options.hydraulic.demand_model = 'PDD'
			return wn
		search_info = dict(search_steps=dict())
		if train_demands=='real':
			train_wn = wn_util.ltown_real()
			# Use thresholds derived from training data for each node
			threshold_file = 'Resources/thresholds.csv'
			thresholds = pd.read_csv(
				threshold_file, index_col='node'
			).squeeze('columns').to_dict()
			search_info['threshold_file'] = 'src/' + threshold_file
		elif train_demands=='toy':
			train_wn = wn_util.ltown_toy()
			thresholds = 1.1
			search_info['threshold'] = thresholds
		else:
			raise AttributeError(
				f"'train_demands' must be either 'toy' or 'real',"
				f" '{train_demands}' is not accepted."
			)
		sensor_path = '../Data/L-Town/pressure_sensors.txt'
		search_info['sensor_config'] = sensor_path
		leakage_detector = BetweenSensorInterpolator(
			nodes_with_sensors = list(np.loadtxt(sensor_path, dtype=str)),
			train_days = 5,
			k = 1,
			thresholds = thresholds
		)
		train_wn.options.hydraulic.demand_model = 'PDD'
		train_wn.options.time.duration = (
			leakage_detector.train_days * SECONDS_PER_DAY
		)
		if not dont_train:
			print('First simulation to train detector...')
			train_pressures = wn_util.compute_pressures(train_wn)
			leakage_detector.train(train_pressures)

		network_file = '../Data/L-Town/L-TOWN_Real.inp'
		leak_duration = 3 * SECONDS_PER_HOUR
		# Constructing a range of potential start times of leaks
		# such that the whole leak fits inside the simulation time.
		# See wn_util.start_time_range for the construction.
		sim_time = 7 * SECONDS_PER_DAY
		wn = wntr.network.WaterNetworkModel(network_file)
		timestep = wn.options.time.hydraulic_timestep
		start_time_range = wn_util.start_time_range(
			start=leakage_detector.train_days * SECONDS_PER_DAY,
			end=sim_time,
			leak_duration=leak_duration,
			timestep=timestep
		)
		lspFinder = cls(
			network_file, leak_duration, leakage_detector, start_time_range,
			network_preparation_method=network_preparation_method,
			search_info=search_info,
			**kwargs
		)
		return lspFinder

	@classmethod
	def hanoi(cls, kind, ignore_nodes=['2', '3']):
		'''Create an LspFinder with default settings for Hanoi.'''
		leakage_detector = BetweenSensorInterpolator(
			nodes_with_sensors = ['4', '13', '16', '22', '31'],
			train_days = 5,
			k = 1,
			thresholds = 0,
			decision_strategy = 'ensemble',
			ensemble_weights = np.load('Resources/best_ensemble_weights.npy')
		)
		network_path = f'../Data/Hanoi_{kind}/'
		train_wn = wntr.network.WaterNetworkModel(network_path + 'train.inp')
		print('First simulation to train detector...')
		train_pressures = wn_util.compute_pressures(train_wn)
		leakage_detector.train(train_pressures)

		leak_duration = 3 * SECONDS_PER_HOUR
		network_file = network_path + 'test.inp'
		wn = wntr.network.WaterNetworkModel(network_file)
		# Constructing a range of potential start times of leaks
		# such that the whole leak fits inside the simulation time.
		# See wn_util.start_time_range for the construction.
		sim_time = wn.options.time.duration
		timestep = wn.options.time.hydraulic_timestep
		start_time_range = wn_util.start_time_range(
			start=0,
			end=sim_time,
			leak_duration=leak_duration,
			timestep=timestep
		)
		sim_start = pd.Timestamp(year=2017, month=7, day=6)
		ignore_nodes = ignore_nodes
		lspFinder = cls(
			network_file, leak_duration, leakage_detector, start_time_range,
			sim_start=sim_start, ignore_nodes=ignore_nodes
		)
		return lspFinder

def run_ltown():
	'''Run the default analysis for L-Town.'''
	lspFinder = LspFinder.ltown()
	lspFinder.find_lsp(algorithm='genetic', verbose=True)

def analyze_ltown_node(node_name, start_time,
		initial_leak_area=0.01, maximization_trials=10, lspFinder=None,
		**kwargs):
	'''
	Maximize the leak area for a node-time pair in L-Town.

	This method uses LspFinder.ltown() (See the code for default settings).

	Parameters
	-----------

	All parameters are passed to LspFinder.maximize_leak_area

	node_name: str
	name of a network junction where the leak should be placed

	start_time: int, start time of the leak in seconds

	initial_leak_area: float, default=0.01 m^2
	an initial guess for the leakage area.

	maximization_trials: int, must be positive, default=10
	number of trials in the leak maximization process

	lspFinder: LspFinder, default=None
	LspFinder to perform the search. If None is given, the normal
	LspFinder.ltown() is used.

	**kwargs: keyword arguments passed to LspFinder.ltown

	Returns
	--------
	max_area: float
	the maximal leak area which does not produce an alarm
	'''
	if lspFinder is None:
		lspFinder = LspFinder.ltown(**kwargs)
	max_area = lspFinder.maximize_leak_area(
		initial_leak_area, node_name, start_time,
		maximization_trials, verbose=True
	)
	return max_area

def run_hanoi(kind):
	'''Run the default analysis for Hanoi.'''
	lspFinder = LspFinder.hanoi()
	lspFinder.find_lsp(algorithm='genetic', verbose=True)

def compare_toy2real(
		train_demands,
		junction_file=None,
		start_time=5*SECONDS_PER_DAY + 5*SECONDS_PER_HOUR + 5*60):
	'''
	Compare the results of area maximization based on training data.

	This method can be used to compare the maximum unnoticed area at some
	nodes of the L-Town network based on whether the leakage detector was
	trained on realistic data or on toy data. In a realistic scenario, one
	might need to train the leakage detector on toy data because realistic
	data is not always available. Depending on the 'train_demands' parameter
	(either 'toy' or 'real') this will save the resulting maximal areas in
	the file 'trained_on_real.npy' or 'trained_on_toy.npy' in the folder
	../Results/L-Town/Detector_Comparison
	If junction names are given in an input file, the corresponding junctions
	are used. Otherwise, 10 junctions are randomly selected.

	Parameters
	-----------

	train_demands: str, 'toy' or 'real'
	Which demands should be used for training?

	junction_file: str, default=None
	path to a '.npy'-file containing junction names in L-Town. If None is
	given, 10 junctions are randomly selected.

	start_time: start time of the leak in seconds
	default: 450300 (5 days, 5 hours and 5 minutes after start)
	The leak will always last three hours.

	This method has no return value.
	'''
	results_path = '../Results/L-Town/Detector_Comparison' 
	if junction_file is None:
		wn = wn_util.ltown_toy()
		junction_names = np.random.choice(
			wn.junction_name_list, size=10, replace=False
		)
		np.save(f'{results_path}/picked_junctions.npy', junction_names)
	else:
		junction_names = np.load(junction_file)
	print('The following junctions were selected:')
	print(junction_names)
	max_areas = []
	lspFinder = LspFinder.ltown(train_demands=train_demands)
	for junction_name in junction_names:
		max_areas.append(
			analyze_ltown_node(junction_name, start_time, lspFinder=lspFinder
		))
	print('The maximal areas were as follows:')
	print(max_areas)
	np.save(
		f'{results_path}/trained_on_{train_demands}.npy',
		np.array(max_areas)
	)

def run_parallel_trials(algorithm, n_trials):
	'''
	Run trials for the genetic or spectral algorithm on L-Town in parallel.

	The following nodes are ignored:
	['n44', 'n111', 'n300', 'n303', 'n336', 'n343']

	The initial area is set to 0.004 m^2.

	Results will be written to '../Results/L-Town/Basic_GA/Trial-<n>' for
	the genetic algorithm and '../Results/L-Town/Spectral_GA/Trial-<n>' for
	the spectral algorithm.

	Parameters
	-----------
	algorithm: one of 'genetic' or 'spectral'
	LSP search algorithm to use

	n_trials: int
	number of parallel trials

	This method has no return value.
	'''
	if algorithm=='genetic':
		performance_paths = [
			f'../Results/L-Town/Basic_GA/Trial-{i}/'
			for i in range(1,n_trials+1)
		]
	elif algorithm=='spectral':
		performance_paths = [
			f'../Results/L-Town/Spectral_GA/Trial-{i}/'
			for i in range(1,n_trials+1)
		]
	ignore_nodes = ['n44', 'n111', 'n300', 'n303', 'n336', 'n343']
	initial_area = 0.004
	def find_lsp_ltown(performance_path):
		lspFinder = LspFinder.ltown(ignore_nodes=ignore_nodes)
		lspFinder.write_search_info(performance_path + 'settings.yaml')
		lspFinder.find_lsp(
			algorithm=algorithm,
			initial_area=initial_area,
			performance_path=performance_path,
			verbose=True
		)
	gen = (
		delayed(find_lsp_ltown)(performance_path)
		for performance_path in performance_paths
	)
	Parallel(n_jobs=n_trials)(gen)

if __name__=='__main__':
	lspFinder = LspFinder.hanoi('2week', ignore_nodes=['2', '3', '20'])
	for i in range(1,6):
		lspFinder.find_lsp(
			algorithm='genetic',
			performance_path=f'../Results/Hanoi_2week/Scenario-2/Basic_GA/Trial-{i}/'
		)
