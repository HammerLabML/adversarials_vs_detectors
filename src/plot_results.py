import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import os
import numpy as np

def compare_results(result_path, max_area_bisection, labels=None):
	'''
	Compare results of different genetic algorithms.

	This will save a figure to {result_path}/result.comparison.png

	Parameters
	----------
	result_path: path (str) containing result folders in the form
	{algorithm_name}/Trial-{n} where algorithm_name is one of Basic_GA,
	Basic_GA or Spectral_GA and n is a positive integer
	starting from 1. The subdirectories must contain files as created by
	wn_util.describe_performance

	max_area_bisection: float, area in m^2
	maximum undetected leak area found by bisection search as a comparison

	labels: dict, labels of the different algorithms in the plot
	default: 
		genetic: basic genetic
		spectral: genetic spectral
	'''
	if labels is None:
		labels = {
			'genetic': 'basic genetic',
			'spectral': 'genetic spectral'
		}
	if result_path[-1]!='/':
		result_path += '/'
	genetic_path = result_path + 'Basic_GA/'
	has_genetic_results	= os.path.isdir(genetic_path)
	spectral_path = result_path + 'Spectral_GA/'
	has_spectral_results = os.path.isdir(spectral_path)
	if not has_genetic_results and not has_spectral_results:
		raise RuntimeError(
			f'Cannot compare results because {result_path}'
			f'does neither contain a subdirectory "Basic_GA", '
			f'nore "Spectral_GA".'
		)
	fig = plt.figure()
	# All y-values are multiplied by 10000 to convert m^2 to cm^2
	plt.axhline(max_area_bisection*10000, label='bisection')
	if has_genetic_results:
		best_areas_genetic = get_best_areas(genetic_path)
		x = list(range(1, len(best_areas_genetic)+1))
		plt.scatter(
			x, best_areas_genetic*10000, label=labels['genetic'], marker='s'
		)
	if has_spectral_results:
		best_areas_spectral = get_best_areas(spectral_path)
		x = list(range(1, len(best_areas_spectral)+1))
		plt.scatter(
			x, best_areas_spectral*10000, label=labels['spectral'], marker='x'
		)
	# Make sure the x-axis uses only integer values
	ax = fig.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.xlabel('Trial')
	plt.ylabel('Maximal Leak Area in cm^2')
	plt.legend()
	plt.savefig(result_path + 'result_comparison.png', dpi=300)

def get_best_areas(path):
	'''
	Retrieve the best leak areas from multiple trials.

	Trials are supposed to be contained in subfolders of results_path in the
	form Trial-<n> where n is a positive integer. See
	wn_util.describe_performance on how to generate the results.

	Parameters
	-----------
	path: path to the directory containing the trials

	Returns
	--------
	maximal leak areas in an np.ndarray. The leak areas are sorted according
	to the trial numbers (Trial-1, Trial-2, ...)
	'''
	subfolders = [f.name for f in os.scandir(path) if f.is_dir()]
	trials = [f for f in subfolders if f.split('-')[0]=='Trial']
	trials = sorted(trials, key=lambda f: int(f.split('-')[-1]))
	trials = [path + trial + '/' for trial in trials]
	best_areas = []
	for trial in trials:
		best_solution_file = trial + 'best_solution.json'
		with open(best_solution_file, 'r') as fp:
			best_solution_dict = json.load(fp)
		best_areas.append(best_solution_dict['area'])
	return np.array(best_areas)

if __name__=='__main__':
	compare_results('../Results/L-Town', 0.008)
#	small_dataset_path = '../Results/Hanoi_1week/'
#	large_dataset_path = '../Results/Hanoi_2week/'
#	print_means(small_dataset_path, large_dataset_path)
