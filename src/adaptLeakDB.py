import wntr
import pandas as pd
import os

from time_constants import SECONDS_PER_HOUR, SECONDS_PER_DAY
import demandGenerator
import wn_util

class LeakDBAdaptor():
	'''
	Class to adapt demand patterns from LeakDB

	To generate an adapted inp-file, use either 'pattern_parsing_workflow'
	to re-construct demand patterns from demand values of a LeakDB scenario
	or 'pattern_generation_workflow' to generate new patterns with the
	demandGenerator script of LeakDB.

	Note: The hydraulic timestep, pattern timestep and report timestep of the
	constructed network will always equal 1800 seconds (half an hour).

	Parameters
	-----------

	settings: dict, properties of the adapted water network
	This must contain:
		sim_time: int, simulation time in seconds
		sim_start: pd.Timestamp object, start of the simulation
		The 'year' field of 'sim_start' must be '2017' as the whole LeakDB
		data lies within this year and the start time must match a half hour
		interval, i.e. the 'minutes' field must be '30' or '0' all fields
		containing smaller time units must be '0'.

	network_file: str
	path to the file containing the original Hanoi network

	Properties
	-----------
	sim_time, sim_start, network_file:
	same values as were passed to the constructor

	sim_end: pd.Timedelta object, time to end the simulation
	This is computed from 'sim_start' and 'sim_time'

	sampling_step: int, 1800 seconds (fixed constant)

	junction_name_list: list of str
	copy of the Hanoi network's junction name list
	'''

	def __init__(self, settings, network_file):
		self.sim_time = settings['sim_time']
		self.sim_start = settings['sim_start']
		self.sim_end = self.sim_start + pd.Timedelta(
			seconds=self.sim_time
		)
		self.sampling_step = 0.5 * SECONDS_PER_HOUR # Like in LeakDB
		self.network_file = network_file

		wn = wntr.network.WaterNetworkModel(network_file)
		self.junction_name_list = wn.junction_name_list

	def set_time_options(self, wn, sim_time=None):
		'''
		Set the time options stored in 'self' for a water network model.

		wn.options.time.duration: see below
		wn.options.time.hydraulic_timestep: self.sampling_step (1800 seconds)
		wn.options.time.pattern_timestep: self.sampling_step (1800 seconds)
		wn.options.time.report_timestep: self.sampling_step (1800 seconds)
		wn.options.quality_timestep: 0

		Parameters
		-----------
		sim_time: int, time in seconds, default=None
		parameter for wn.options.time.duraiton. If none is given,
		self.sim_time is used
		'''
		if sim_time is None:
			sim_time = self.sim_time
		wn.options.time.duration = sim_time
		wn.options.time.hydraulic_timestep = self.sampling_step
		wn.options.time.quality_timestep = 0 # I don't analyze water quality
		wn.options.time.report_timestep = self.sampling_step
		wn.options.time.pattern_timestep = self.sampling_step
		return wn

	def parse_demands(self, demand_path):
		'''
		Parse demand values from multiple csv-files into a single DataFrame.

		Note: The original demand values are converted from m^3/h to m^3/s and
		returned in that form in order to comply with the wntr unit system.

		Parameters
		-----------

		demand_path: str, path to a folder
		the folder must contain csv-files with the following structure:
		name: 'Node_<i>.csv' where '<i>' is replaced by a junction name for
			each junction of the Hanoi network
		header line: Timestamp,Value
		entries:
			"Timestamp": objects that can be converted to pandas.DateTimeIndex
			"Value": demand values in m^3/h (cubic meter per hour)

		Returns
		--------
		A pandas.DataFrame containing demand values for each node in the Hanoi
		network in m^3/s.
		'''
		demands = []
		for junction_name in self.junction_name_list:
			demand = pd.read_csv(
				f'{demand_path}/Node_{junction_name}.csv',
				parse_dates=['Timestamp'], index_col='Timestamp'
			)
			demand.rename(
				columns={'Value': f'Node_{junction_name}'}, inplace=True
			)
			demands.append(demand)
		# Combine demands in single DataFrame
		demands = pd.concat(demands, axis=1)
		# Convert back to m^3 / s
		demands /= SECONDS_PER_HOUR
		return demands

	def demands_to_patterns(self, demands):
		'''
		Convert demand values (m^3/s) to multipliers of base demands.

		Note: I use the term 'demand' to refer to volumetric flow rates while
		I use 'pattern' to refer to (dimensionless) multipliers of demands.

		The base demands for each node are retrieved from self.network_file.

		Parameters
		-----------

		demands: pandas.DataFrame
		demand values in m^3/s

		Returns
		--------
		A pandas.DataFrame containing multipliers for each node's base demand.
		'''
		wn = wntr.network.WaterNetworkModel(self.network_file)
		base_demands = pd.Series(
			{f'Node_{junction_name}': wn.get_node(junction_name).base_demand\
			for junction_name in self.junction_name_list}
		)
		patterns = demands / base_demands
		return patterns

	def generate_patterns(self):
		'''
		Generate patterns and return them in a DataFrame.

		This is a wrapper for demandGenerator.genDem from LeakDB.

		Returns
		--------
		A pandas.DataFrame containing patterns for each node in the
		Hanoi network.
		'''
		patterns = {}
		for junction_name in self.junction_name_list:
			patterns[f'Node_{junction_name}'] = demandGenerator.genDem()
		timestamp = pd.date_range(
			start='2017-01-01 00:00',
			end='2017-12-31 23:30',
			freq='30min'
		)
		patterns = pd.DataFrame(patterns, index=timestamp)
		return patterns

	def wn_from_prepared_patterns(self, patterns, sim_time=None):
		'''
		Create a copy of the Hanoi network from prepared patterns.

		This method is used inside self.wn_from_patterns and
		self.train_test_wns. In contrast to these methods, it requires the
		pattern matrix to span only over the actual simulation time of a
		network and it returns a network rather than writing .inp-files.

		Parameters
		-----------

		patterns: pandas.DataFrame
		This must contain dimensionless multipliers for
		each node's base demand, as returned by 'generate_patterns'.
		The index is expected to be a pandas.DateTimeIndex
		spanning the whole simulation time of the network to be created in
		half-hour intervals and the column names must have the form 'Node_<i>'
		where '<i>' is replaced by a junction name for each junction in the
		Hanoi network.

		sim_time: int, time in seconds, optional
		the simulation time of the netowrk which is passed to
		self.set_time_options. If none is given, self.sim_time is used.

		Returns
		--------
		wn: wntr.network.WaterNetworkModel
		a copy of the Hanoi network equipped with the given patterns and
		specific time options (see self.set_time_options).
		'''
		wn = wntr.network.WaterNetworkModel()
		for junction_name in self.junction_name_list:
			wn.add_pattern(
				f'Pattern_{junction_name}',
				patterns[f'Node_{junction_name}'].to_list()
			)
		# So far, wn only contains patterns.
		# Now, the Hanoi network architecture is added
		fileReader = wntr.epanet.io.InpFile()
		wn = fileReader.read(self.network_file, wn)
		wn = self.set_time_options(wn, sim_time=sim_time)
		for junction_name in self.junction_name_list:
			node = wn.get_node(junction_name)
			node.add_demand(node.base_demand, f'Pattern_{junction_name}')
			# Delete the constant demand from the inp file
			node.demand_timeseries_list.pop(0)
		return wn

	def wn_from_patterns(self, full_year_patterns, kind):
		'''
		Create a copy of the Hanoi network with given patterns.

		For the simulation parameters concerning time, see
		'self.set_time_options'.

		Note: This method has no Return-value. Instead, it will save the
		created network to a file called "Hanoi_<month>_<kind>.inp" where
		'<month>' is replaced by the month of self.sim_start and
		'<kind>' is replaced by the 'kind'-parameter.

		Parameters
		-----------

		full_year_patterns: pandas.DataFrame
		This must contain dimensionless multipliers for
		each node's base demand, as returned by 'generate_patterns'.
		The index is expected to be a pandas.DateTimeIndex
		spanning the whole year 2017 in half-hour intervals and the column
		names must have the form 'Node_<i>' where '<i>' is replaced by a
		junction name for each junction in the Hanoi network.

		kind: str, 'fixed' or 'generated'
		This is used only for the name of the generated inp-file to indicate
		how the demands were constructed.
		'''
		if kind not in ['fixed', 'generated']:
			raise ValueError("'kind' must be one of 'fixed' or 'generated'")
		sim_patterns = full_year_patterns[self.sim_start:self.sim_end]
		wn = self.wn_from_prepared_patterns(sim_patterns)
		start_month = self.sim_start.strftime("%B")
		wn.write_inpfile(f'Hanoi_{start_month}_{kind}.inp')

	def train_test_wns(self, full_year_patterns, train_days, kind):
		'''
		Create a train and test network from successive patterns.

		This will create 2 .inp-files and a folder for them (if it does not
		already exist). Both networks will be copies of the Hanoi network,
		differing only in patterns and total simulation time. The folder will
		be named "Hanoi_<month>_<kind>.inp" where '<month>' is replaced by the
		month of self.sim_start and '<kind>' is replaced by the
		'kind'-parameter. The networks inside the folder will be named
		'train.inp' and 'test.inp'. The simulation of the test network will
		start immediately one timestep after the end of the simulation of the
		train network. This is achieved by cutting out successive chunks from
		the 'full_year_patterns' matrix. The sum of simulation times of train
		and test network is given by self.sim_time. See self.set_time_options
		for further time options.

		Parameters
		-----------

		full_year_patterns: pandas.DataFrame
		This must contain dimensionless multipliers for
		each node's base demand, as returned by 'generate_patterns'.
		The index is expected to be a pandas.DateTimeIndex
		spanning the whole year 2017 in half-hour intervals and the column
		names must have the form 'Node_<i>' where '<i>' is replaced by a
		junction name for each junction in the Hanoi network.

		train_days: int
		starting at self.sim_start, this amount of days of full_year_pressures
		is used for the training network, while the reamining part until
		self.sim_end is used for the test netowrk.
		Note that (like in numpy slicing) the simulation step at
		self.sim_time + train_days * SECONDS_PER_DAY
		is included only in the test network.

		kind: str, 'fixed' or 'generated'
		This is used only for the name of the generated inp-file to indicate
		how the demands were constructed.
		'''
		if kind not in ['fixed', 'generated']:
			raise ValueError("'kind' must be one of 'fixed' or 'generated'")
		# I need to subtract 1 because pandas slices include the upper end
		train_time = train_days * SECONDS_PER_DAY - 1
		train_end = self.sim_start + pd.Timedelta(seconds=train_time)
		train_patterns = full_year_patterns[self.sim_start:train_end]
		train_wn = self.wn_from_prepared_patterns(
			train_patterns, sim_time=train_time
		)

		test_start = train_end + pd.Timedelta(seconds=1)
		test_patterns = full_year_patterns[test_start:self.sim_end]
		test_time = (self.sim_end - test_start).total_seconds()
		test_wn = self.wn_from_prepared_patterns(
			test_patterns, sim_time=test_time
		)

		start_month = self.sim_start.strftime("%B")
		dir_name = f'Hanoi_{start_month}_{kind}/'
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		train_wn.write_inpfile(dir_name + 'train.inp')
		test_wn.write_inpfile(dir_name + 'test.inp')
		return

	def pattern_generation_workflow(self):
		'''
		Create a copy of the Hanoi-network with freshly generated patterns.

		This method uses the time options stored in 'self' for the wn and
		applies the demandGenerator from LeakDB to create the patterns. The
		resulting network is saved to the file 'Hanoi_<month>_generated.inp'
		where '<month>' is replaced by the month of self.sim_start.
		'''
		full_year_patterns = self.generate_patterns()
		self.wn_from_patterns(full_year_patterns, 'generated')

	def pattern_parsing_workflow(self):
		'''
		Create a copy of the Hanoi network with re-constructed patterns.

		The patterns are created from demand values stored in LeakDB scenario
		1. These demands are compared to the base-demands of each node to
		reconstruct the multipliers. This method uses time options stored in
		'self' for the wn.  The resulting network is stored in a file called
		'Hanoi_<month>_fixed.inp' where '<month>' is replaced by the month of
		self.sim_start.
		'''
		demand_path = '../Data/LeakDB_Example_Demands'
		full_year_demands = self.parse_demands(demand_path)
		full_year_patterns = self.demands_to_patterns(full_year_demands)
		self.wn_from_patterns(full_year_patterns, 'fixed')

if __name__=='__main__':
	settings = dict(
		sim_time = 14 * SECONDS_PER_DAY,
		sim_start = pd.Timestamp(year=2017, month=6, day=24)
	)
	network_file = '../Network_Originals/Hanoi_CMH.inp'
	leakDBAdaptor = LeakDBAdaptor(settings, network_file)
#	full_year_patterns = leakDBAdaptor.generate_patterns()
#	leakDBAdaptor.train_test_wns(full_year_patterns, 5, 'generated')
	leakDBAdaptor.pattern_generation_workflow()
	
