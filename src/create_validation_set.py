import wntr
from numpy import random
import sys

from LeakProperties import LeakProperties
import wn_util
from time_constants import SECONDS_PER_HOUR, SECONDS_PER_DAY

def create_validation_set(network_file, start_time_range, area, duration, output_path, ignore_nodes=[]):
	'''
	Create a dataset for threshold validation from an adapted network.

	This method will read in a WaterNetworkModel, place a leak in one of its
	junctions and save information about the leak and pressure values of the
	models simulation to two different files. The leak will be placed at a
	random junction and with a random starting time, but with the given area
	and duration.

	Note: No other parameters of the network are changed here, only the leak
	is added. Hence, the network and in particular its time options must
	already be prepared.

	Parameters
	-----------

	network_file: str
	path to an EPANET inp-file

	start_time_range: list-like
	this should contain all allowed leak starting times and take into account
	that some of the network's pressure values are used in the training phase
	of the leakage detector. During this training time, no leak should be
	added.

	area: float
	area of the leak in m^2

	duration: int
	duration of the leak in seconds

	output_path: str
	path to the folder the output files should be written to

	ignore_nodes: list of str
	leaks will not be placed at these nodes

	The method will create the following files in the folder given by
	output_path:

	leak_info.json: properties of the leak. This can be read into a
	LeakProperties object by calling
	LeakProperties.from_json(output_path + "leak_info.json")

	pressures.csv: pressure values of a network simulation after the leak was
	added
	'''
	wn = wntr.network.WaterNetworkModel(network_file)
	junction_names = wn.junction_name_list
	for ignore_node in ignore_nodes:
		junction_names.remove(ignore_node)
	junction_name = random.choice(junction_names)
	start_time = random.choice(start_time_range)
	leak_properties = LeakProperties(
		area=area,
		junction_name=junction_name,
		start_time=start_time,
		duration=duration
	)

	if output_path[-1] != '/':
		output_path += '/'
	leak_properties.to_json(output_path + 'leak_info.json')
	pressures = wn_util.compute_pressures(wn, leak_properties=leak_properties)
	pressures.to_csv(output_path + 'pressures.csv', index_label='time')

if __name__=='__main__':
	if len(sys.argv) != 3:
		print(
			f'Please specify a network file and an output path'
			f' as command-line arguments!'
		)
		exit()
	network_file = sys.argv[1]
	area = 0.0075
	duration = 3 * SECONDS_PER_HOUR
	start_time_range = wn_util.start_time_range(
		start = 5 * SECONDS_PER_DAY,
		end = 14 *SECONDS_PER_DAY,
		leak_duration = duration,
		timestep = 0.5 * SECONDS_PER_HOUR
	)
	output_path = sys.argv[2]
	create_validation_set(
		network_file, start_time_range, area, duration, output_path,
		ignore_nodes = ['2', '3']
	)
