import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.optimize import minimize, minimize_scalar, Bounds

from time_constants import SECONDS_PER_DAY
from LeakProperties import LeakProperties
from LeakageDetectors import SingleSensorForecaster, BetweenSensorInterpolator

def time_to_detection_score(alarm_times, leak_properties):
	'''
	Compute a score for the time it takes to detect a leak.

	The score is returned on a 0-1 scale, where 1 means that the leak was
	detected immediately and 0 means that it was not detected, which could
	also mean that an alarm was raised only after the leak had already been
	fixed. All intermediate values are computed as 1 - relative_ttd, where
	relative_ttd is a quotient of the actual time before the leak was detected
	and the leak's duration.

	Parameters
	-----------
	alarm_times: array-like
	times of alarms in seconds

	leak_properties: LeakProperties.LeakProperties object
	this is used to retrieve the leak start and duration

	Returns
	--------
	score between 0 (bad) and 1 (perfect)
	'''
	start = leak_properties.start_time
	alarm_delays = [alarm - start for alarm in alarm_times if alarm >= start]
	if not alarm_delays:
		return 0
	total_ttd = min(alarm_delays)
	relative_ttd = total_ttd / leak_properties.duration
	return 1 - relative_ttd if relative_ttd < 1 else 0

class ThresholdValidator():
	'''
	Tool to test alarm thresholds of a given LeakageDetector

	Parameters
	-----------

	leakage_detector: an instance of a subclass of AbstractLeakageDetector
	A threshold can be tested for this leakage_detector. For this purpose, the
	threshold of the given instance will be overwritten.

	scenario_paths: list of str
	paths to folders each of which must contain the following files:
	leak_info.json: information about a leak that was placed in a
	WaterNetworkModel
	pressures.csv: pressure values from a simulation of the same model
	These files can be constructed using create_validation_set.py.
	'''

	def __init__(self, leakage_detector, scenario_paths):
		self.leakage_detector = leakage_detector
		self.scenario_paths = scenario_paths

	def score_for_scenario(self, scenario_path, verbose=False):
		'''
		Uses self.leakage_detector to compute a score for scenario_path.

		The score is determined by the harmonic mean of the
		time to detection score and the precision:
		2 * ttd_score * precision / (ttd_score + precision)

		Parameters
		-----------

		scenario_path: str
		a directory containing a file with simulated pressure values
		(pressures.csv) and a file with leak properties used for the
		simulation (leak_info.json).

		verbose: bool, default=False
		if true, the scenario path as well as the single scores (time to
		detection and precision) will be printed.
		'''
		if scenario_path[-1] != '/':
			scenario_path += '/'
		if verbose:
			print(f'Path: {scenario_path}')
		pressure_file = scenario_path + 'pressures.csv'
		pressures = pd.read_csv(pressure_file, index_col='time')
		alarm_times = self.leakage_detector.train_and_detect(pressures)
		# Getting times at which alarms might be generated
		test_start = self.leakage_detector.train_days * SECONDS_PER_DAY
		test_times = pressures.loc[test_start:].index
		has_alarm = [time in alarm_times for time in test_times]

		leak_file = scenario_path + 'leak_info.json'
		leak_properties = LeakProperties.from_json(leak_file)
		leak_start = leak_properties.start_time
		leak_end = leak_start + leak_properties.duration
		leak_check = lambda time: time >= leak_start and time < leak_end
		has_leak = [leak_check(time) for time in test_times]

		ttd_score = time_to_detection_score(alarm_times, leak_properties)
		if verbose:
			print(f'Time To Detection Score: {ttd_score:.3f}')

		conf_mat = confusion_matrix(has_leak, has_alarm)
		false_positives, true_positives = conf_mat[:,1]
		if false_positives==0 and true_positives==0:
			prec = 0
		else:
			prec = true_positives / (true_positives + false_positives)
		if verbose:
			print(f'Precision: {prec:.3f}')

		if prec==0 and ttd_score==0:
			harmonic_mean = 0
		else:
			harmonic_mean = 2 * prec * ttd_score / (prec + ttd_score)
		if verbose:
			print(f'Score: {harmonic_mean:.3f}')
		return harmonic_mean

	def validate(self, thresholds, **kwargs):
		'''
		Evaluates the quality of a set of thresholds.

		This method computes self.score_for_scenario for all scenarios in
		self.scenario_path after setting the thresholds for
		self.leakage_detector. For the method to compute the score, see the
		documentation of self.score_for_scenario.

		Parameters
		-----------

		thresholds: array-like
		the thresholds to be validated

		**kwargs: Keyword arguments passed to self.score_for_scenario

		Returns
		---------
		scores: dict, keys=sceanrio paths, values=scores
		The resulting scores for each scenario
		'''
		self.leakage_detector.thresholds = dict(
			zip(self.leakage_detector.nodes_with_sensors, thresholds)
		)
		scores = {
			sp: self.score_for_scenario(sp, **kwargs)
			for sp in self.scenario_paths
		}
		return scores

	def validate_ensemble(self, ensemble_weights, **kwargs):
		'''
		Evaluates the quality of a set of ensemble weights.

		This method computes self.score_for_scenario for all scenarios in
		self.scenario_path after setting the ensemble weights for
		self.leakage_detector. For the method to compute the score, see the
		documentation of self.score_for_scenario.

		Parameters
		-----------

		ensemble_weights: array-like
		the ensemble weights to be validated

		**kwargs: Keyword arguments passed to self.score_for_scenario

		Returns
		---------
		scores: dict, keys=sceanrio paths, values=scores
		The resulting scores for each scenario
		'''
		self.leakage_detector.ensemble_weights = dict(
			zip(self.leakage_detector.nodes_with_sensors, ensemble_weights)
		)
		scores = {
			sp: self.score_for_scenario(sp, **kwargs)
			for sp in self.scenario_paths
		}
		return scores

	def validate_single_node(self, threshold, observed_node_name, **kwargs):
		'''
		Validate the threshold for a single node.

		This method helps to observe the performance gained by a single
		threshold sepeartely. In order to do this, thresholds for all other
		nodes are set to a too high value (100 in this case) so they won't
		trigger an alarm. The paths in self.scenario_paths are used for the
		validation

		Parameters
		-----------

		threshold: float
		threshold to be set for the observed node

		observed_node_name: str

		**kwargs: keyword-arguments passed to self.score_for_scenario

		Returns
		---------
		scores: dict, keys=sceanrio paths, values=scores
		The resulting scores for each scenario
		'''
		# Set a too high threshold for all other nodes
		set_threshold = lambda node_name: threshold if node_name==observed_node_name else 100
		self.leakage_detector.thresholds = {
			node_name: set_threshold(node_name)
			for node_name in self.leakage_detector.nodes_with_sensors
		}
		scores = {
			sp: self.score_for_scenario(sp, **kwargs)
			for sp in self.scenario_paths
		}
		return scores

	def negative_average_score(self, thresholds):
		'''
		Computes the negative average of all scores of given thresholds (see
		self.validate). This is useful as input to optimizers.
		'''
		scores = self.validate(thresholds)
		average_score = np.array(list(scores.values())).mean()
		return average_score * (-1)

	def negative_average_score_single_node(
			self, threshold, observed_node_name):
		'''
		Computes the negative average of all scores of a threshold for one
		observed node (see self.validate_single_node). This is useful as input
		to optimizers.
		'''
		scores = self.validate_single_node(threshold, observed_node_name)
		average_score = np.array(list(scores.values())).mean()
		res = average_score * (-1)
		return res

	def negative_average_score_ensemble(self, ensemble_weights):
		'''
		Computes the negative average of all scores of given ensemble weights
		(see self.validate_ensemble). This is useful as input to optimizers.
		'''
		scores = self.validate_ensemble(ensemble_weights)
		average_score = np.array(list(scores.values())).mean()
		return average_score * (-1)

	@classmethod
	def example(cls):
		'''
		Example threshold validator

		This validator uses a BetweenSensorInterpolator to predict leakages.
		The thresholds of this leakage detector are initially set to zero and
		expected to be changed by the user later on (e.g. by the
		validate-method).  See the source code for detailed configuration.
		'''
		leakage_detector = BetweenSensorInterpolator(
			nodes_with_sensors = ['4', '13', '16', '22', '31'],
			train_days = 7,
			k = 1,
			thresholds = 0
		)
		scenario_paths = []
		n_scenarios = 3
		first_scenario = 7
		for i in range(first_scenario, n_scenarios + first_scenario):
			scenario_paths.append(
				f'../Data/Threshold_Validation/Scenario-{i}/'
			)
		tv = cls(leakage_detector, scenario_paths)
		return tv

	@classmethod
	def ensemble_example(cls):
		'''
		Example threshold validator for ensemble weight optimization.

		This validator uses a BetweenSensorInterpolator with
		decision_strategy='ensemble' to predict leakages.  The ensemble
		weights of this leakage detector are initially set to zero and
		expected to be changed by the user later on (e.g. by the
		validate_ensemble-method). Thresholds are not used and also set to
		zero. See the source code for detailed configuration.
		'''
		nodes_with_sensors = ['4', '13', '16', '22', '31']
		leakage_detector = BetweenSensorInterpolator(
			nodes_with_sensors = nodes_with_sensors,
			train_days = 5,
			k = 1,
			thresholds = 0,
			decision_strategy = 'ensemble',
			ensemble_weights = {node: 0 for node in nodes_with_sensors}
		)
		scenario_paths = [
			f'../Data/Ensemble_Validation/Scenario-{i}'
			for i in range(1,21)
		]
		tv = cls(leakage_detector, scenario_paths)
		return tv

def optimize_thresholds_jointly():
	'''
	Optimize all thresholds of the example validator jointly.

	Joint optimization is achieved by the scipy implementation of the Powell
	optimization method. This method will print the optimal thresholds, the
	score they achieved (including the single scores for every scenario of the
	example validator) and the optimization steps taken to reach these
	thresholds in order to diagnose problems with the optimizer.
	'''
	tv = ThresholdValidator.example()
	n_sensors = len(tv.leakage_detector.nodes_with_sensors)
	optimizer_options = {'return_all': True}
	optimization_result = minimize(
		tv.negative_average_score, x0=0.562 * np.ones(n_sensors),
		method='Powell', options=optimizer_options
	)
	best_thresholds = optimization_result.x
	best_average_score = optimization_result.fun * (-1)
	print(
		f'The optimal thresholds {np.around(best_thresholds, decimals=3)}'
		f' achieved an average score of {best_average_score:.3f}!'
	)
	print('The intermediate optimization results were as follows:')
	for vec in optimization_result.allvecs:
		print(vec)
	print('The scores for the single paths were as follows:')
	scores = tv.validate(best_thresholds, verbose=True)

def optimize_single_threshold(observed_node_name, upper_bound):
	'''
	Optimize the threshold for a single node.

	This method uses the scipy implementation of a bounded-search
	optimization. It will print the best threshold achieved, scores for each
	scenario of the example validator and an average of the single scores.

	Parameters:
	------------

	observed_node_name: str

	upper_bound: float
	upper bound for the threshold (lower bound is 0)

	Returns
	--------
	the best average score achieved by the optimization
	'''
	tv = ThresholdValidator.example()
	optimization_result = minimize_scalar(
		tv.negative_average_score_single_node, method='bounded',
		args=(observed_node_name), bounds=(0,upper_bound)
	)
	best_threshold = optimization_result.x
	best_average_score = optimization_result.fun * (-1)
	print(
		f'The optimal threshold {best_threshold:.3f}'
		f' for node {observed_node_name}'
		f' achieved an average score of {best_average_score:.3f}!'
	)
	print('The scores for the single paths were as follows:')
	scores = tv.validate_single_node(
		best_threshold, observed_node_name, verbose=True
	)
	return best_average_score

def optimize_single_threshold_iteratively(
		observed_node_name, initial_upper_bound):
	'''
	Run optimize_single_node until the final score is satisfactory.

	This will run optimize_single_node with the given node and half the upper
	bound if the search resulted in an average score < 0.25. This procedure is
	repeated until the score exceeds 0.25

	Parameters
	-----------

	node: str
	the observed node for which the threshold should be optimized

	initial_upper_bound:
	the upper bound used in the first run of optimize_single_node (e.g. 20)
	'''
	upper_bound = initial_upper_bound
	res = optimize_single_threshold(observed_node_name, upper_bound)
	n_runs = 1
	while (res < 0.2):
		upper_bound /= 2
		res = optimize_single_threshold(observed_node_name, upper_bound)
		n_runs += 1
	print(f'Number of runs: {n_runs}')

def optimize_ensemble_weights():
	'''
	Optimize the ensemble weights of the example ensemble validator.

	Optimization is achieved by the scipy implementation of the Powell
	optimization method. This method will print the optimal ensemble weights
	and the score they achieved (including the single scores for every
	scenario of the validator).

	This method will save results to a file 'best_ensemble_weights.npy' which
	can be read by calling numpy.load('best_ensemble_weights.npy')
	'''
	tv = ThresholdValidator.ensemble_example()
	n_sensors = len(tv.leakage_detector.nodes_with_sensors)
	optimization_result = minimize(
		tv.negative_average_score_ensemble, x0=0.8 * np.ones(n_sensors),
		method='Powell'
	)
	best_ensemble_weights = optimization_result.x
	print('The scores for the single paths were as follows:')
	scores = tv.validate(best_ensemble_weights, verbose=True)
	best_average_score = optimization_result.fun * (-1)
	print(
		f'The optimal ensemble_weights'
		f' {np.around(best_ensemble_weights, decimals=3)}'
		f' achieved an average score of {best_average_score:.3f}!'
	)
	np.save('best_ensemble_weights.npy', best_ensemble_weights)

if __name__=='__main__':
	optimize_ensemble_weights()
