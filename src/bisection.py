def bisection_step(var, upwards, lower_bound, upper_bound, downward_scale=2, upward_scale=2):
	''' Helper for bisection_search '''
	if upwards:
		if upper_bound is None:
			var = var * upward_scale
		else:
			var = (var + upper_bound) / 2
	else: # if downwards
		if lower_bound is None:
			var = var / downward_scale
		else:
			var = (var + lower_bound) / 2
	return var

def bisection_search(func, sought_result, start_value, n_trials,
		lower_bound=None, upper_bound=None,
		step_parameters={'downward_scale': 2, 'upward_scale': 2},
		meta_info=False, **kwargs):
	'''
	Use binary search to find a function's inverse.

	Important: This works only for monotoniously increasing functions
	which map positive inputs to positive outputs.

	Parameters
	-----------

	func: function

	sought_result: numeric
	function value, for which an inverse is sought.

	start_value: numeric
	initial guess for the inverse

	n_trials: int
	trials to approximate the inverse. If it cannot be determined exactly,
	the best approximation is returned.

	lower_bound, upper_bound: numeric, optional, default=None
	initial lower and upper bound for the search space

	step_parameters: dict, optional,
	default={'downward_scale': 2, 'upward_scale': 2}
	determines the behaviour of the function as long as bounds are not
	determined yet: By default, if a step upwards is taken and no upper bound
	is present, the previous result is multiplied by 2. This can be changed
	with the upward scale parameter. Likewise, if a step downwards is taken
	and no lower bound is present yet, the result is divided by 2. This can be
	changed by the downward_scale parameter.

	meta_info: bool, optional, default: False
	Does the function return a tuple? If true, the second tuple element
	for the inverse is returned as well.

	**kwargs: optional
	key-word arguments passed to func in every run
	
	Returns
	--------
	x: numeric
	s.t. func(x) = sought_result (at least approximately)

	bounds: tuple of numbers
	upper and lower bound for x

	info:
	If meta_info=True and
	func(x) = sought_result, info
	then info is also returned
	'''
	trials = 0
	current = start_value
	while True:
		result = func(current, **kwargs)
		if meta_info:
			# We assume the function to return some additional
			# information, so we have to perform tuple unpacking
			result, info = result
		if result < sought_result:
			previous = current
			current = bisection_step(
				current, True, lower_bound, upper_bound, **step_parameters
			)
			lower_bound = previous
		elif result > sought_result:
			previous = current
			current = bisection_step(
				current, False, lower_bound, upper_bound, **step_parameters
			)
			upper_bound = previous
		trials += 1
		if result == sought_result or trials >= n_trials:
			break
	bounds = lower_bound, upper_bound
	# if the function produced additional information,
	# we want to include it in the return value
	if meta_info:
		return current, bounds, info
	return current, bounds

def mock_alarms_for_area(x):
	''' Toy function '''
	if x <= 0.005:
		return 0
	elif x < 0.006:
		return 1
	else:
		return 3

if __name__=='__main__':
	print(
		bisection_search(
			mock_alarms_for_area, 1, 0.01, 10,
			step_parameters={'upward_scale': 1.2, 'downward_scale': 1.2}
		)
	)

