from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from time_constants import SECONDS_PER_DAY, HOURS_PER_DAY
import wn_util

def train_test_split(X, y, split_index):
	'''
	Split predictors and targets at some index.

	Parameters
	-----------

	X, y: np.ndarray
	classification predictor variables and target variables, respectively

	split_index: The index at which X and y should be split

	Returns
	--------
	X_train, X_test, y_train, y_test: np.ndarray
	the training predictors, test predictors, training targets and
	test targets, respectively
	'''
	X_train, X_test, y_train, y_test = (
		X[:split_index], X[split_index:],
		y[:split_index], y[split_index:]
	)
	return X_train, X_test, y_train, y_test

class AbstractLeakageDetector():
	'''
	This class is not meant to be instantiated directly.

	Please see the documentation of its childrens constructors for help.
	'''

	def __init__(self, nodes_with_sensors, train_days, k, thresholds,
			  decision_strategy, ensemble_weights, detector_type):
		self.nodes_with_sensors = nodes_with_sensors
		self.train_days = train_days
		self.k = k
		self.thresholds = thresholds
		allowed_strategies = ['compliant_with_each', 'ensemble']
		if decision_strategy not in allowed_strategies:
			raise AttributeError(
				f"'decision_strategy' must be one of {allowed_methods}"
			)
		self.decision_strategy = decision_strategy
		if decision_strategy=='ensemble':
			if ensemble_weights is None:
				raise ValueError(
					f'You have to pass ensemble weights'
					f' for an ensemble decision strategy!'
				)
		self.ensemble_weights = ensemble_weights
		allowed_detector_types = [
			'Single Sensor Forecaster',
			'Between Sensor Interpolator',
			'Between Sensor Forecaster'
		]
		if detector_type not in allowed_detector_types:
			raise ValueError(
				f"'detector_type' must be one of {allowed_detector_types}"
			)
		self.type = detector_type

	def __repr__(self):
		res = (
			f'Type: {self.type}\n'
			f'Nodes with sensors: {self.nodes_with_sensors}\n'
			f'Train days: {self.train_days}\n'
			f'k: {self.k}\n'
			f'Thresholds: {self.thresholds}\n'
			f'Decision strategy: {self.decision_strategy}\n'
		)
		if self.ensemble_weights:
			rounded = {
				k: np.round(v, 3) for k,v in self.ensemble_weights.items()
			}
			res += f'Ensemble Weights: {rounded}'
		else:
			res += f'Ensemble Weights: {self.ensemble_weights}'
		return res

	def _match(self, values):
		'''Helper method for initialization'''
		if type(values)==dict or values is None:
			return values
		else:
			try: # is it iterable?
				return dict(zip(self.nodes_with_sensors, values))
			except TypeError: # or atomic?
				matched = {
					node_name: values for node_name in self.nodes_with_sensors
				}
				return matched

	@property
	def thresholds(self):
		return self._thresholds

	@thresholds.setter
	def thresholds(self, thresholds):
		self._thresholds = self._match(thresholds)

	@property
	def ensemble_weights(self):
		return self._ensemble_weights

	@ensemble_weights.setter
	def ensemble_weights(self, ensemble_weights):
		self._ensemble_weights = self._match(ensemble_weights)

	def alarm_times(self, residuals, method=None):
		'''
		Generate an alarm based on residuals.

		Parameters
		-----------
		residuals: pd.DataFrame, rows=timesteps, columns=nodes
		differences between the detectors prediction and the actual pressure
		values at self.nodes_with_sensors

		method: str, method to raise the alarm,
		default=self.decision_strategy
		current options are:
			- compliant_with_each: the residuals at a given timestep must be
			  compliant with (i.e. smaller than) EVERY sensor threshold. If
			  the residuals exceed the threshold for at least one sensor, an
			  alarm is raised
			- ensemble: the residuals for each node are weighted by
			  self.ensemble_weights[node]. These weighted residuals are than
			  summed up and an alarm is raised if the sum is greater than 1.
			  Hyperparameter validation should be performed for the ensemble
			  weights. Note: self.thresholds are not taken into account in
			  this method.

		Returns
		--------
		an array containing the times in seconds since the simulation start
		at which an alarm was raised. An empty array corresponds to no alarm
		'''
		if method is None:
			method = self.decision_strategy
		if method=='compliant_with_each':
			exceeded = (residuals > pd.Series(self.thresholds))
			any_exceeded = exceeded.any(axis=1)
			# Get indices where 'any_exceeded' is True
			alarm_times = any_exceeded[any_exceeded].index
		elif method=='ensemble':
			weighted_residuals = residuals * pd.Series(self.ensemble_weights)
			alarm_score = weighted_residuals.sum(axis=1)
			alarm_times = weighted_residuals[alarm_score > 1].index
		return alarm_times

	def set_overlap_pressures(self, pressures, overlap_steps):
		'''
		Store the last pressure values used for training.

		This is useful if one uses previous timesteps and wants to predict
		some values immediately after the training period. The last few
		pressure values from the training set can then later be re-used for
		the prediction of the first few test values.

		Parameters
		----------

		pressures: pd.DataFrame
		training pressure values

		overlap_steps: int
		how many of the last pressure values should be saved

		This sets self.overlap_pressures
		'''
		timestep = pressures.index[1] - pressures.index[0]
		last_measuring_time = pressures.index[-1]
		overlap_pressures = pressures.tail(overlap_steps)
		overlap_pressures.index -= (last_measuring_time + timestep)
		self.overlap_pressures = overlap_pressures

class SingleSensorForecaster(AbstractLeakageDetector):
	'''
	Linear Forecaster for pressure values based on a fixed number of timesteps

	For M different nodes and a fixed number of timesteps k, this detecter
	uses M linear models which each use the last k timesteps for their
	corresponding node in order to predict the next timestep.

	Parameters
	-----------

	nodes_with_sensors: list of str
	the names of junctions in a wntr.network.WaterNetworkModel that should be
	equipped with virtual pressure sensors.

	train_days: int
	this is used in self.train_and_detect to determine how many of the rows of
	a given pressure matrix should be used for training the models. The actual
	leakage detection will be performed on the remaining part.
	For example, for a water network with pressure measurements
	every 30 minutes, train_days=2 would mean that the first 48*2=96 rows of
	the pressure matrix are used for training.

	k: int
	number of timesteps to use for pressure detecting

	thresholds: dict, iterable or numeric, must be positive
	the thresholds for the generation of alarms. If a dict is specified, the
	keys must be equal to nodes_with_sensors. Other iterable objects are
	zipped with nodes_with_sensors for a matching. A single atomic threshold
	will be assigned to all sensors.
	The unit of this is meter. (pressure = meter * gravity * waterdensity)

	decision_strategy: str, one of 'compliant_with_each' or 'ensemble'
	default: 'compliant_with_each'
	This determines under which conditions an alarm should be raised based on
	the residuals (differences between actual and measured pressure values)
	and the thresholds. See the 'method' parameter in the documentation of
	self.alarm_times for the effect of the decision strategy. In the default
	case, an alarm is raised if the residual for at least one sensor exceeds
	the sensor threshold.

	ensemble_weights: dict, default=None,
	required only if decision_strategy='ensemble'
	weights for the residuals of each node to reflect their importance in the
	alarm decision. If decision_strategy=='ensemble', the residual matrix for
	each timestep is weighted with ensemble_weights and summed across the
	nodes. An alarm is raised if the sum exceeds 1.
	Note: thresholds are ignored in this case.
	'''

	def __init__(self, nodes_with_sensors, train_days, k, thresholds,
			  decision_strategy='compliant_with_each', ensemble_weights=None):
		super().__init__(
			nodes_with_sensors, train_days, k, thresholds,
			decision_strategy, ensemble_weights, 'Single Sensor Forecaster'
		)
		self.scalers = dict()
		self.models = dict()

	def k_step_predictors(self, X):
		'''
		Split columns of X into k-shingles.

		This method is used to construct predictors
		for self.detect.

		Parameters
		-----------

		X: np.ndarray
		matrix of shape N X M

		Returns
		--------
		A List with M elements. Each element is an np.ndarray
		with shape N-self.k x self.k

		Note:
		The last elements of each column are not returned, as they are
		only targets and no predictors.
		'''
		N, M = X.shape
		res = []
		for m in range(M):
			k_steps = np.row_stack(
				[X[n:n+self.k, m] for n in range(N-self.k)]
			)
			res.append(k_steps)
		return res

	def k_step_targets(self, X):
		'''
		Give shortened columns of X as a list, dropping the k first elements.
		'''
		N, M = X.shape
		return [X[self.k:N,m] for m in range(M)]

	def train(self, pressures, set_overlap=False):
		'''
		Learn linear predictions for self.nodes_with_sensors.

		For M different nodes, this method trains M linear models which use
		the last self.k timesteps for their corresponding node in order to
		predict the next timestep. The predictors are normalized with a
		separate StandardScaler for each model. The scalers are stored in
		self.scalers and the models are stored in self.models.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		set_overlap: bool, optional, default=False
		if True, this will cause the last self.k rows of pressures to be
		stored in self.overlap_pressures. These can later be utilized by
		setting use_overlap=True in self.detect. This has the advantage that a
		leak starting at the very first timestep after training can already be
		found.
		
		This method sets self.models and self.scalers, but it has no return
		value.
		'''
		if set_overlap:
			self.set_overlap_pressures(pressures, self.k)
		pressures = pressures[self.nodes_with_sensors].to_numpy()
		Xs = self.k_step_predictors(pressures)
		ys = self.k_step_targets(pressures)
		for X, y, node_name in zip(Xs, ys, self.nodes_with_sensors):
			scaler = StandardScaler()
			scaler.fit(X)
			self.scalers[node_name] = scaler
			X = scaler.transform(X)
			model = LinearRegression()
			model.fit(X, y)
			self.models[node_name] = model

	def compute_residuals(self, pressures, use_overlap=False):
		'''
		Compare pressure predictions to observed values.

		Note: self.scalers and self.models must be filled by self.train before
		this method can be used.

		For each node name in self.nodes_with_sensors self.models[node_name]
		is used to predict its pressure values after the predictors
		(pressure values from previous timesteps) have been scaled with
		self.scalers[node_name]. The positive difference between predicted
		and observed pressure values is returned.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		use_overlap: bool, optional, default=False
		if True and if set_overlap has been set to true in self.train,
		self.overlap pressures are prepended to pressures before starting the
		leakage detection. The index values of pressures remain in tact.
		self.overlap pressures will get matching negative timesteps as
		indices. use_overlap=True has the advantage that the last elements of
		the training set can already be used for forecasting during the first
		couple of timesteps.

		Returns
		--------
		a pandas.DataFrame with residuals
		index: seconds since simulation start
		columns: self.nodes_with_sensors
		'''
		if use_overlap:
			try:
				pressures = pd.concat((self.overlap_pressures, pressures))
			except AttributeError:
				print(
					f'You cannot use overlap pressures '
					f'if none were set during training!'
				)
				exit()
		timesteps = pressures.index
		pressures = pressures[self.nodes_with_sensors].to_numpy()
		Xs = self.k_step_predictors(pressures)
		ys = self.k_step_targets(pressures)
		# Empty residuals dataframe to be filled later
		residuals = pd.DataFrame(
			index=timesteps,
			columns=self.nodes_with_sensors
		)
		# Remove timesteps without prediction
		residuals = residuals[self.k:]
		for X, y, node_name in zip(Xs, ys, self.nodes_with_sensors):
			scaler = self.scalers[node_name]
			X = scaler.transform(X)
			model = self.models[node_name]
			residuals[node_name] = np.abs(model.predict(X) - y)
		return residuals

	def detect(self, pressures, **kwargs):
		'''Detect leaks by applying self.alarm_times to pressure residuals.'''
		residuals = self.compute_residuals(pressures, **kwargs)
		return self.alarm_times(residuals)

	def train_and_detect(self, pressures):
		'''
		Train pressure forecasting models and use them for leakage detection.

		self.train_days determines how many rows of the pressure
		matrix are used to train the models. Leakage detection is performed on
		the remaining part.

		Note: There is an overlap between training and test set because the
		models always require the k previous timesteps for the prediction.
		Hence, the last k rows of the training set will also be included in
		the test set to detect leaks starting at the very beginning of the
		test set.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		Returns
		--------
		an array containing the times in seconds since the simulation start
		at which an alarm was raised. An empty array corresponds to no alarm
		'''
		# I need to subtract 1 because pandas slices include the upper end
		train_seconds = self.train_days * SECONDS_PER_DAY - 1
		self.train(pressures.loc[:train_seconds])
		offset = self.k * (pressures.index[1] - pressures.index[0])
		alarm_times = self.detect(pressures.loc[train_seconds-offset:])
		return alarm_times

	@classmethod
	def example(cls):
		''' Create an example Single Sensor Forecaster'''
		nodes_with_sensors = ['4', '13', '16', '22', '31']
		train_days = 5
		k = 4
		threshold = 1.5
		example = cls(
			nodes_with_sensors, train_days, k, threshold
		)
		return example

class BetweenSensorInterpolator(AbstractLeakageDetector):
	'''
	Linear interpolator of pressure values between observed nodes

	For each of the given nodes, this will build a model to predict the
	pressure values at that node based on the pressure values of all the other
	nodes. Hence, for M nodes, M different models will be trained. Each of the
	models receives an M-1 dimensional input vector and outputs a single
	scalar value at each timestep.

	Important: As opposed to the SingleSensorForecaster, this detector does
	not predict values that may occur in the future. Rather, it tries to
	interpolate values that occured at other nodes, but within the same
	timestep.

	One may specify a window size k to average the input pressure values to
	the model with the pressure values of the k-1 previous timesteps. See
	"Adversarial Attacks and Robustness in Water Distribution Systems" for a
	formal description.

	Parameters
	-----------

	nodes_with_sensors: list of str, junction names
	observed nodes used to construct the linear models

	train_days: int
	this is used in self.train_and_detect to determine how many of the rows of
	a given pressure matrix should be used for training the models. The actual
	leakage detection will be performed on the remaining part.
	For example, for a water network with pressure measurements
	every 30 minutes, train_days=2 would mean that the first 48*2=96 rows of
	the pressure matrix are used for training.

	k: int, number of timesteps to average for the classificaiton input
	use 1 to take only the current timestep into account

	thresholds: dict, iterable or numeric, must be positive
	the thresholds for the generation of alarms. If a dict is specified, the
	keys must be equal to nodes_with_sensors. Other iterable objects are
	zipped with nodes_with_sensors for a matching. A single atomic threshold
	will be assigned to all sensors.
	The unit of this is meter. (pressure = meter * gravity * waterdensity)

	decision_strategy: str, one of 'compliant_with_each' or 'ensemble'
	default: 'compliant_with_each'
	This determines under which conditions an alarm should be raised based on
	the residuals (differences between actual and measured pressure values)
	and the thresholds. See the 'method' parameter in the documentation of
	self.alarm_times for the effect of the decision strategy. In the default
	case, an alarm is raised if the residual for at least one sensor exceeds
	the sensor threshold.

	ensemble_weights: dict, default=None,
	required only if decision_strategy='ensemble'
	weights for the residuals of each node to reflect their importance in the
	alarm decision. If decision_strategy=='ensemble', the residual matrix for
	each timestep is weighted with ensemble_weights and summed across the
	nodes. An alarm is raised if the sum exceeds 1.
	Note: thresholds are ignored in this case.
	'''

	def __init__(self, nodes_with_sensors, train_days, k, thresholds,
			  decision_strategy='compliant_with_each', ensemble_weights=None):
		super().__init__(
			nodes_with_sensors, train_days, k, thresholds,
			decision_strategy, ensemble_weights, 'Between Sensor Interpolator'
		)
		self.scalers = dict()
		self.models = dict()

	def k_step_averages(self, X):
		'''Average self.k successive rows of a matrix together.'''
		N = X.shape[0]
		res = np.row_stack(
			[X[n:n+self.k, :].mean(axis=0) for n in range(N-self.k+1)]
		)
		return res

	def train(self, pressures, set_overlap=False):
		'''
		Learn a linear interpolation between self.nodes_with_sensors.

		For M different nodes, this will create M linear models, each of which
		tries to predict the pressure value of one node based on the pressure
		values of the other nodes. Predictor and prediction generally belong
		to the same timestep, unless the predictors were averaged with
		previous timesteps (see self.k). After the averaging (if any) the
		predictors are normalized with a separate StandardScaler for each
		model. The scalers are stored in self.scalers and the models are
		stored in self.models. Note that self.scaler[my_node_name] yields a
		scaler that was used to scale pressure values of all nodes except
		my_node. 

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		set_overlap: bool, optional, default=False
		if True and if self.k > 1, this will cause the last self.k-1 rows of
		pressures to be stored in self.overlap_pressures. These can later be
		utilized by setting use_overlap=True in self.detect. This has the
		advantage that a leak starting at the very first timestep after
		training can already be found, even when avering over previous
		timesteps.
		
		This method sets self.models and self.scalers, but it has no return
		value.
		'''
		if set_overlap:
			assert self.k > 1, (
				f'You cannot set overlap pressures because no overlap is used'
				f' in this detector!'
			)
			self.set_overlap_pressures(pressures, self.k-1)
		pressures = pressures[self.nodes_with_sensors]
		for predicted_node_name in self.nodes_with_sensors:
			X = pressures.drop(columns=[predicted_node_name]).to_numpy()
			if self.k > 1:
				X = self.k_step_averages(X)
			scaler = StandardScaler()
			scaler.fit(X)
			self.scalers[predicted_node_name] = scaler
			X = scaler.transform(X)
			y = pressures[predicted_node_name].to_numpy()
			y = y[self.k-1:]
			model = LinearRegression()
			model.fit(X, y)
			self.models[predicted_node_name] = model

	def compute_residuals(self, pressures, use_overlap=False):
		'''
		Compare pressure predictions to observed values.

		Note: self.scalers and self.models must be filled by self.train before
		this method can be used.

		For each node name in self.nodes_with_sensors self.models[node_name]
		is used to predict its pressure values after the predictors (pressure
		values from other nodes) have been scaled with
		self.scalers[node_name]. The positive difference between these
		predicted pressure values and observed pressures is returned.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		use_overlap: bool, optional, default=False
		if True and if set_overlap has been set to True in self.train,
		self.overlap pressures are prepended to pressures before starting the
		leakage detection. The index values of pressures remain in tact.
		self.overlap pressures will get matching negative timesteps as
		indices. use_overlap=True has the advantage that the last elements of
		the training set can already be used for averaging during the first
		couple of timesteps.

		Returns
		--------
		a pandas.DataFrame with residuals
		index: seconds since simulation start
		columns: self.nodes_with_sensors
		'''
		if not (self.models and self.scalers):
			raise RuntimeError('You cannot detect leakages before training!')
		if use_overlap:
			try:
				pressures = pd.concat((self.overlap_pressures, pressures))
			except AttributeError:
				print(
					f'You cannot use overlap pressures '
					f'if none were set during training!'
				)
				exit()
		timesteps = pressures.index
		pressures = pressures[self.nodes_with_sensors]
		# Empty residuals dataframe to be filled later
		residuals = pd.DataFrame(
			index=timesteps,
			columns=self.nodes_with_sensors
		)
		# Remove timesteps without prediction
		residuals = residuals[self.k-1:]
		for predicted_node_name in self.nodes_with_sensors:
			X = pressures.drop(columns=[predicted_node_name]).to_numpy()
			if self.k > 1:
				X = self.k_step_averages(X)
			scaler = self.scalers[predicted_node_name]
			X = scaler.transform(X)
			y = pressures[predicted_node_name].to_numpy()
			y = y[self.k-1:]
			model = self.models[predicted_node_name]
			residuals[predicted_node_name] = np.abs(model.predict(X) - y)
		return residuals

	def detect(self, pressures, **kwargs):
		'''Detect leaks by applying self.alarm_times to pressure residuals.'''
		residuals = self.compute_residuals(pressures, **kwargs)
		return self.alarm_times(residuals)

	def train_and_detect(self, pressures):
		'''
		Train linear interpolation models and use them for leakage detection.

		self.train_days determines how many rows of the pressure
		matrix are used to train the models. Leakage detection is performed on
		the remaining part.

		Note: If self.k > 1, there is an overlap between training and test set
		because the models always require the k-1 previous timesteps for
		averaging. Hence, the last k-1 rows of the training set will also be
		included in the test set to detect leaks starting at the very
		beginning of the test set.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		Returns
		--------
		an array containing the times in seconds since the simulation start
		at which an alarm was raised. An empty array corresponds to no alarm
		'''
		# I need to subtract 1 because pandas slices include the upper end
		train_seconds = self.train_days * SECONDS_PER_DAY - 1
		self.train(pressures.loc[:train_seconds])
		overlap = (self.k-1) * (pressures.index[1] - pressures.index[0])
		alarm_times = self.detect(pressures.loc[train_seconds-overlap:])
		return alarm_times

	@classmethod
	def example(cls):
		''' Create an example Between Sensor Interpolator (Hanoi)'''
		nodes_with_sensors = ['4', '13', '16', '22', '31']
		train_days = 5
		k = 1
		threshold = 0.562
		example = cls(
			nodes_with_sensors, train_days, k, threshold
		)
		return example

	@classmethod
	def ensemble_example(cls):
		'''
		Create an example Between Sensor Interpolator with ensemble
		decision strategy (Hanoi).
		'''
		nodes_with_sensors = ['4', '13', '16', '22', '31']
		train_days = 5
		k = 1
		threshold = 0.562
		decision_strategy = 'ensemble'
		ensemble_weights = np.load('Resources/best_ensemble_weights.npy')
		example = cls(
			nodes_with_sensors, train_days, k, threshold,
			decision_strategy, ensemble_weights
		)
		return example

class BetweenSensorForecaster(AbstractLeakageDetector):
	'''
	Like the BetweenSensorInterpolator, but using the previous timestep(s)
	for the prediction. How many timeteps are used is given by the
	hyperparameter k.
	'''

	def __init__(self, nodes_with_sensors, train_days, k, thresholds,
			  decision_strategy='compliant_with_each', ensemble_weights=None):
		super().__init__(
			nodes_with_sensors, train_days, k, thresholds,
			decision_strategy, ensemble_weights, 'Between Sensor Forecaster'
		)
		self.scalers = dict()
		self.models = dict()

	def k_step_averages(self, X):
		'''Average self.k successive rows of a matrix together.'''
		N = X.shape[0]
		res = np.row_stack(
			[X[n:n+self.k, :].mean(axis=0) for n in range(N-self.k+1)]
		)
		return res

	def train(self, pressures):
		'''
		Learn a linear forecasting between self.nodes_with_sensors.

		For M different nodes, this will create M linear models, each of which
		tries to predict the pressure value of one node based on the pressure
		values of the other nodes and from the previous timestep and
		potentially further timesteps. The hyperparameter self.k determines
		how many timesteps are used. If k>1, pressure values from earlier
		timesteps are averaged with pressure values from the previous
		timestep. After the averaging (if any) the predictors are normalized
		with a separate StandardScaler for each model. The scalers are
		stored in self.scalers and the models are stored in self.models.
		Note that self.scaler[my_node_name] yields a scaler that was used to
		scale pressure values of all nodes EXCEPT my_node. 

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values
		
		This method sets self.models and self.scalers, but it has no return
		value.
		'''
		pressures = pressures[self.nodes_with_sensors]
		for predicted_node_name in self.nodes_with_sensors:
			X = pressures.drop(columns=[predicted_node_name]).to_numpy()
			# Drop the last row because there is no later pressure
			# to be predicted
			X = X[:-1, :]
			if self.k > 1:
				X = self.k_step_averages(X)
			scaler = StandardScaler()
			scaler.fit(X)
			self.scalers[predicted_node_name] = scaler
			X = scaler.transform(X)
			y = pressures[predicted_node_name].to_numpy()
			y = y[self.k:]
			model = LinearRegression()
			model.fit(X, y)
			self.models[predicted_node_name] = model

	def compute_residuals(self, pressures):
		'''
		Compare pressure predictions to observed values.

		Note: self.scalers and self.models must be filled by self.train before
		this method can be used.

		For each node name in self.nodes_with_sensors self.models[node_name]
		is used to predict its pressure values after the predictors
		(pressure values from other nodes and previous timestep(s)) have
		been scaled with self.scalers[node_name]. The positive difference
		between these predicted pressure values and observed pressures is
		returned.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		Returns
		--------
		a pandas.DataFrame with residuals
		index: seconds since simulation start
		columns: self.nodes_with_sensors
		'''
		if not (self.models and self.scalers):
			raise RuntimeError('You cannot detect leakages before training!')
		timesteps = pressures.index
		pressures = pressures[self.nodes_with_sensors]
		# Empty residuals dataframe to be filled later
		residuals = pd.DataFrame(
			index=timesteps,
			columns=self.nodes_with_sensors
		)
		# Remove timesteps without prediction
		residuals = residuals[self.k:]
		for predicted_node_name in self.nodes_with_sensors:
			X = pressures.drop(columns=[predicted_node_name]).to_numpy()
			X = X[:-1, : ]
			if self.k > 1:
				X = self.k_step_averages(X)
			scaler = self.scalers[predicted_node_name]
			X = scaler.transform(X)
			y = pressures[predicted_node_name].to_numpy()
			y = y[self.k:]
			model = self.models[predicted_node_name]
			residuals[predicted_node_name] = np.abs(model.predict(X) - y)
		return residuals

	def detect(self, pressures):
		'''Detect leaks by applying self.alarm_times to pressure residuals.'''
		residuals = self.compute_residuals(pressures)
		return self.alarm_times(residuals)

