import json
from time_constants import SECONDS_PER_HOUR

class LeakProperties():
	'''
	Properties of a leak in a water network

	Parameters
	-----------

	area: float
	area of the leak in m^2

	junction_name: str
	name of a junction in in the network where the leak should be placed

	start_time: int
	time in seconds

	duration: int
	time from the start of the leak until it is fixed, given in seconds
	'''

	def __init__(self, area, junction_name, start_time, duration):
		self.area = float(area)
		self.junction_name = str(junction_name)
		self.start_time = int(start_time)
		self.duration = int(duration)

	def to_dict(self):
		'''Convert to a dictionary with keys for every attribute.'''
		dict_representation = dict(
			area=self.area,
			junction_name=self.junction_name,
			start_time=self.start_time,
			duration=self.duration
		)
		return dict_representation

	def to_json(self, output_file):
		'''Write the dictionary representation to "output_file" in json format.'''
		with open(output_file, 'w') as fp:
			json.dump(self.to_dict(), fp)

	@classmethod
	def from_json(cls, leakfile):
		'''Load a LeakProperties object from a json-file.'''
		with open(leakfile, 'r') as fp:
			dict_representation = json.load(fp)
		return cls(**dict_representation)

	@classmethod
	def example(cls):
		'''Example LeakProperties'''
		example = cls(
			area = 0.01,
			junction_name = '6',
			start_time = 10 * SECONDS_PER_HOUR,
			duration = 3 * SECONDS_PER_HOUR
		)
		return example

