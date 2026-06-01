import warnings
import itertools

import pandas as pd
import numpy as np
import math as math

import matplotlib.pyplot as plt
import netCDF4 as nc
import pandapower as pp

from . import config
from . import utils
from .const import *
from . import engine

from . import fragilitycurve
from . import hazard

# For stratification using the R language
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.rlike.container as rlc

from mpl_toolkits.basemap import Basemap

# TODO: improve warning messages in Network and PowerElement
# TODO: displace fragility curve elements
# TODO: revise following contants
# OBS: Not all elements can be set to i_montecarlo = True. The montecarlo_database.csv file cannot be empty

COL_NAME_FRAGILITY_CURVE = 'fragilityCurve'
COL_NAME_KF = 'kf'
COL_NAME_RESILIENCE_FULL = 'resilienceFull'
COL_NAME_WEATHER_TTR = 'weatherTTR'
TIMESERIES_CLASS = pd.Series

# Network element status
STATUS = {'on': 1, 'off': 0, 'reparing': -1, 'waiting': -2}

def build_class_dict(df, class_name):
	return {row.id: globals()[class_name](row.dropna(axis=0).to_dict()) for index, row in utils.df_to_internal_fields(df).iterrows()}


def find_element_in_standard_dict_list(id, list):
	# return first element or None otherwise
	return next((x for x in list if 'id' in x and x['id'] == id), None)


def standard_dict(content, type, id, field):
	return {k: v for k, v in locals().items()}


def get_datatype_elements(object, class_):
	if isinstance(object, list):
		iterateOver = object
		out = []
	elif isinstance(object, dict):
		iterateOver = object.values()
		out = []
	elif hasattr(object, "__dict__"):
		iterateOver = vars(object).values()
		# out = [(object, type(object).__name__, key) for key, value in vars(object).items() if isinstance(value, class_)]
		out = [standard_dict(content=object, type=type(object).__name__, id=object.id, field=key)
			   for key, value in vars(object).items() if isinstance(value, class_)]
	else:
		iterateOver = []
		out = []
	return out + list(itertools.chain(*[get_datatype_elements(x, class_) for x in iterateOver]))

def build_database(standard_dict_list, get_value_from_content=True):
	columnNames = [*standard_dict_list[0]]
	columnNames.remove('content')
	formatted_dict = {}
	for d in standard_dict_list:
		value = d['content']
		if get_value_from_content:
			value = getattr(value, d['field'])
		formatted_dict[(*[d[x] for x in columnNames],)] = value
	out = pd.DataFrame.from_dict(formatted_dict)
	out.columns.names = columnNames
	return out


class Network:
	'''
	TODO: @TIM add description
	Add description of Newtork class here
	'''

	def __init__(self, simulationName):
		self.id = None
		self.f_hz = None
		self.sn_mva = None

		self.totalInstalledPower = 0
		self.totalConventionalPower = 0
		self.totalRenewablePower = 0
		self.totalStoragePower = 0

		self.nodes = {}
		self.generators = {}
		self.externalGenerators = {}
		self.loads = {}
		self.transformers = {}
		self.transformerTypes = {}
		self.lines = {}
		self.lineTypes = {}
		self.switches = {}
		self.crews = {}

		self.fragilityCurves = fragilitycurve.build_fragility_curve_database(simulationName)
		self.event = hazard.Hazard(simulationName)
		self.returnPeriods = hazard.build_return_period_database(simulationName)

		self.pp_network = None

		self.outagesSchedule = None
		self.crewSchedule = None
		self.switchesSchedule = None

		self.metrics = []
		self.mcVariables = []

		self.build_network(config.path.networkFile(simulationName))
		self.calculationEngine = engine.pandapower(self.pp_network)

		self.powerElements = {**self.lines,
								**self.generators,
								**self.loads,
								**self.transformers,
								**self.nodes}
		print(f"Network for study case <{simulationName}> initialized.")

	def build_network_parameters(self, networkFile):
		df_network = pd.read_excel(networkFile, sheet_name=SHEET_NAME_NETWORK)
		for index, row in utils.df_to_internal_fields(df_network).iterrows():
			for key, value in row.dropna(axis=0).to_dict().items():
				if hasattr(self, key):
					setattr(self, key, value)
				else:
					warnings.warn(f'Input parameter "{key}" unknown in network parameters.')
		return df_network

	def build_nodes(self, networkFile):
		df_nodes = pd.read_excel(networkFile, sheet_name=SHEET_NAME_NODES)
		self.nodes = build_class_dict(df_nodes, 'Bus')
		return df_nodes

	def build_generators(self, networkFile):
		df_gen = pd.read_excel(networkFile, sheet_name=SHEET_NAME_GENERATORS)
		df_ex_gen = pd.read_excel(
			networkFile, sheet_name=SHEET_NAME_EXTERNAL_GEN)
		self.generators = build_class_dict(df_gen, 'Generator')
		self.externalGenerators = build_class_dict(df_ex_gen, 'Generator')
		return df_gen, df_ex_gen

	def build_loads(self, networkFile):
		df_load = pd.read_excel(networkFile, sheet_name=SHEET_NAME_LOADS)
		self.loads = build_class_dict(df_load, 'Load')
		return df_load

	def build_transformers(self, networkFile):
		df_transformers = pd.read_excel(
			networkFile, sheet_name=SHEET_NAME_TRANSFORMER)
		df_tr_types = pd.read_excel(networkFile, sheet_name=SHEET_NAME_TR_TYPE)
		self.transformers = build_class_dict(df_transformers, 'Transformer')
		self.transformerTypes = build_class_dict(df_tr_types, 'Transformer')
		return df_transformers, df_tr_types

	def build_lines(self, networkFile):
		df_lines = pd.read_excel(networkFile, sheet_name=SHEET_NAME_LINES)
		df_ln_types = pd.read_excel(networkFile, sheet_name=SHEET_NAME_LN_TYPE)
		self.lines = build_class_dict(df_lines, 'Line')
		self.lineTypes = build_class_dict(df_ln_types, 'Line')
		return df_lines, df_ln_types

	def build_switches(self, networkFile):
		df_switches = pd.read_excel(
			networkFile, sheet_name=SHEET_NAME_SWITCHES)
		self.switches = build_class_dict(df_switches, 'Switch')
		return df_switches

	def build_crews(self, networkFile):
		df_crews = pd.read_excel(networkFile, sheet_name=SHEET_NAME_CREWS)
		self.crews = build_class_dict(df_crews, 'Crew')

	def allocate_profiles(self, networkFile):
		df_profiles = pd.read_excel(
			networkFile, sheet_name=SHEET_NAME_PROFILES, header=[0, 1], index_col=0)
		df_profiles.reset_index(drop=True, inplace=True)
		for assetId, field in df_profiles:
			element = self.get_powerelement(assetId)
			if hasattr(element, field):  # element can be None depending on get_powerelement()
				setattr(element, field, df_profiles[(assetId, field)])
			else:
				warnings.warn(f'Missing "{field}" field or "{assetId}" asset during profile allocation.')

	def build_network(self, networkFile, build_pp_network=True):
		df_network = self.build_network_parameters(networkFile)
		df_nodes = self.build_nodes(networkFile)
		df_load = self.build_loads(networkFile)
		df_gen, df_ex_gen = self.build_generators(networkFile)
		df_transformers, df_tr_types = self.build_transformers(networkFile)
		df_lines, df_ln_types = self.build_lines(networkFile)
		df_switches = self.build_switches(networkFile)
		df_cost = pd.read_excel(
			networkFile, sheet_name=SHEET_NAME_COST)
		self.build_crews(networkFile)
		self.allocate_profiles(networkFile)
		if build_pp_network:
			self.pp_network = self.build_pp_network(
				df_network=df_network,
				df_bus=df_nodes,
				df_tr=df_transformers,
				df_tr_type=df_tr_types,
				df_ln=df_lines,
				df_ln_type=df_ln_types,
				df_load=df_load,
				df_ex_gen=df_ex_gen,
				df_gen=df_gen,
				df_switch=df_switches,
				df_cost=df_cost
			)

	def update_failure_probability(self, intensity=None, ref_return_period=None):
		'''
		This function updates the failure probability of the power elements of the network.
		The event object must be defined before calling this method.
		If this method is called the failure probabilities entered in the excel file will not be considered.
		'''
		for el in self.powerElements.values():
			el.update_failure_probability(self, intensity=intensity, ref_return_period=ref_return_period)
		return self.powerElements

	def build_pp_network(self, df_network, df_bus, df_tr, df_tr_type, df_ln, df_ln_type, df_load, df_ex_gen, df_gen, df_switch, df_cost):
		# TODO: it seems this funciton is missplaced. Can it be moved to engine.pandapower?
		# TODO: Condense creation of dictionaries by iteration
		# TODO: update fields_map.csv accordingly
		'''
		This Function takes as imput the input file name without the extention
		and gives as output the pandapower network object.
		'''

		df_network = df_network.replace({np.nan: None})
		name = df_network[COL_NAME_NAME].values[0]
		f_hz = df_network[COL_NAME_FREQUENCY].values[0]
		sn_mva = df_network[COL_NAME_REF_POWER].values[0]
		kwargs_network = dict(f_hz=f_hz, name=name, sn_mva=sn_mva)
		network = pp.create_empty_network(
			**{key: value for key, value in kwargs_network.items() if value is not None})

		# Creating the bus elements
		df_bus = df_bus.replace({np.nan: None})
		bus_ids = {}
		for index, row in df_bus.iterrows():
			if (row[COL_NAME_LONGITUDE] is not None) and (row[COL_NAME_LATITUDE] is not None):
				geodata = (row[COL_NAME_LATITUDE], row[COL_NAME_LONGITUDE])
			else:
				geodata = None
			kwargs_bus = dict(net=network, vn_kv=row[COL_NAME_VOLTAGE],
							  name=row[COL_NAME_NAME],
							  index=row[COL_NAME_INDEX],
							  geodata=geodata, zone=row[COL_NAME_ZONE],
							  in_service=row[COL_NAME_SERVICE],
							  max_vm_pu=row[COL_NAME_MAX_VOLTAGE],
							  min_vm_pu=row[COL_NAME_MIN_VOLTAGE])

			bus_ids[row[COL_NAME_NAME]] = pp.create_bus(
				**{key: value for key, value in kwargs_bus.items() if value is not None})

		# Creating the transformer elements
		df_tr = df_tr.replace({np.nan: None})
		df_tr_type = df_tr_type.replace({np.nan: None})
		tr_ids = {}
		for index, row in df_tr.iterrows():
			tr_type = df_tr_type.loc[df_tr_type[COL_NAME_NAME]
									 == row[COL_NAME_TYPE]]
			kwargs_tr = dict(net=network, hv_bus=bus_ids[row[COL_NAME_NODE_P]],
							 lv_bus=bus_ids[row[COL_NAME_NODE_S]],
							 sn_mva=tr_type[COL_NAME_REF_POWER].values[0],
							 vn_hv_kv=tr_type[COL_NAME_VN_HV].values[0],
							 vn_lv_kv=tr_type[COL_NAME_VN_LV].values[0],
							 vkr_percent=tr_type[COL_NAME_VKR].values[0],
							 vk_percent=tr_type[COL_NAME_VK].values[0],
							 pfe_kw=tr_type[COL_NAME_PFE].values[0],
							 i0_percent=tr_type[COL_NAME_I0].values[0],
							 shift_degree=tr_type[COL_NAME_SHIFT].values[0],
							 tap_side=tr_type[COL_NAME_TAP_SIDE].values[0],
							 tap_neutral=tr_type[COL_NAME_TAP_NEUTRAL].values[0],
							 tap_max=tr_type[COL_NAME_TAP_MAX].values[0],
							 tap_min=tr_type[COL_NAME_TAP_MIN].values[0],
							 tap_step_percent=tr_type[COL_NAME_TAP_STEP].values[0],
							 tap_step_degree=tr_type[COL_NAME_TAP_STEP_ANGLE].values[0],
							 tap_pos=row[COL_NAME_TAP_POS],
							 tap_phase_shifter=tr_type[COL_NAME_TAP_PHASE_SHIFTER].values[0],
							 in_service=row[COL_NAME_SERVICE],
							 name=row[COL_NAME_NAME],
							 vector_group=row[COL_NAME_VECTOR_GROUP],
							 max_loading_percent=row[COL_NAME_MAX_LOADING],
							 parallel=row[COL_NAME_PARALLEL],
							 df=row[COL_NAME_DF])
			tr_ids[row[COL_NAME_NAME]] = pp.create_transformer_from_parameters(
				**{key: value for key, value in kwargs_tr.items() if value is not None})

		# Creating the line elements
		df_ln = df_ln.replace({np.nan: None})
		df_ln_type = df_ln_type.replace({np.nan: None})
		line_ids = {}
		for index, row in df_ln.iterrows():
			if (row[COL_NAME_FROM_LONGITUDE] is not None) and (row[COL_NAME_FROM_LATITUDE] is not None) and (row[COL_NAME_TO_LONGITUDE] is not None) and (row[COL_NAME_TO_LATITUDE] is not None):
				geodata = [[row[COL_NAME_FROM_LATITUDE], row[COL_NAME_FROM_LONGITUDE]],
						   [row[COL_NAME_TO_LATITUDE], row[COL_NAME_TO_LONGITUDE]]]
			else:
				geodata = None
			ln_type = df_ln_type.loc[df_ln_type[COL_NAME_NAME]
									 == row[COL_NAME_TYPE]]
			kwargs_ln = dict(net=network,
							 from_bus=bus_ids[row[COL_NAME_FROM_BUS]],
							 to_bus=bus_ids[row[COL_NAME_TO_BUS]],
							 length_km=row[COL_NAME_LENGTH],
							 r_ohm_per_km=ln_type[COL_NAME_R1].values[0],
							 x_ohm_per_km=ln_type[COL_NAME_X1].values[0],
							 c_nf_per_km=ln_type[COL_NAME_C1].values[0],
							 max_i_ka=ln_type[COL_NAME_MAX_I].values[0],
							 name=row[COL_NAME_NAME],
							 index=row[COL_NAME_INDEX],
							 type=ln_type[COL_NAME_TYPE].values[0],
							 geodata=geodata,
							 in_service=row[COL_NAME_SERVICE],
							 df=row[COL_NAME_DF],
							 parallel=row[COL_NAME_PARALLEL],
							 g_us_per_km=ln_type[COL_NAME_G1].values[0],
							 max_loading_percent=row[COL_NAME_MAX_LOADING],
							 r0_ohm_per_km=ln_type[COL_NAME_R0].values[0],
							 x0_ohm_per_km=ln_type[COL_NAME_X0].values[0],
							 c0_nf_per_km=ln_type[COL_NAME_C0].values[0],
							 g0_us_per_km=ln_type[COL_NAME_G0].values[0])
			line_ids[row[COL_NAME_NAME]] = pp.create_line_from_parameters(
								**{key: value for key, value in kwargs_ln.items() if value is not None})

		# Creating the load elements
		df_load = df_load.replace({np.nan: None})
		load_ids = {}
		for index, row in df_load.iterrows():
			kwargs_load = dict(net=network,
							   bus=bus_ids[row[COL_NAME_BUS]],
							   p_mw=row[COL_NAME_P],
							   q_mvar=row[COL_NAME_Q],
							   const_z_percent=row[COL_NAME_CONST_Z],
							   const_i_percent=row[COL_NAME_CONST_I],
							   sn_mva=row[COL_NAME_REF_POWER],
							   name=row[COL_NAME_NAME],
							   scaling=row[COL_NAME_SCALING],
							   in_service=row[COL_NAME_SERVICE],
							   type=row[COL_NAME_TYPE],
							   max_p_mw=row[COL_NAME_MAX_P],
							   min_p_mw=row[COL_NAME_MIN_P],
							   max_q_mvar=row[COL_NAME_MAX_Q],
							   min_q_mvar=row[COL_NAME_MIN_Q],
							   controllable=row[COL_NAME_CONTROLLABLE])
			load_ids[row[COL_NAME_NAME]] = pp.create_load(
				**{key: value for key, value in kwargs_load.items() if value is not None})

		# Creating the external gen elements
		df_ex_gen = df_ex_gen.replace({np.nan: None})
		ex_gen_ids = {}
		for index, row in df_ex_gen.iterrows():
			kwargs_ex_gen = dict(net=network,
								 bus=bus_ids[row[COL_NAME_BUS]],
								 vm_pu=row[COL_NAME_VM],
								 va_degree=row[COL_NAME_VA],
								 name=row[COL_NAME_NAME],
								 in_service=row[COL_NAME_SERVICE],
								 s_sc_max_mva=row[COL_NAME_MAX_S_SC],
								 s_sc_min_mva=row[COL_NAME_MIN_S_SC],
								 rx_max=row[COL_NAME_MAX_RX],
								 rx_min=row[COL_NAME_MIN_RX],
								 max_p_mw=row[COL_NAME_MAX_P],
								 min_p_mw=row[COL_NAME_MIN_P],
								 max_q_mvar=row[COL_NAME_MAX_Q],
								 min_q_mvar=row[COL_NAME_MIN_Q],
								 r0x0_max=row[COL_NAME_MAX_R0X0],
								 x0x_max=row[COL_NAME_MAX_X0X],
								 controllable=row[COL_NAME_CONTROLLABLE],
								 slack_weight=row[COL_NAME_SLACK_WEIGHT])
			ex_gen_ids[row[COL_NAME_NAME]] = pp.create_ext_grid(
				**{key: value for key, value in kwargs_ex_gen.items() if value is not None})

		# Creating the generator elements
		df_gen = df_gen.replace({np.nan: None})
		gen_ids = {}
		for index, row in df_gen.iterrows():
			kwargs_gen = dict(net=network,
							  bus=bus_ids[row[COL_NAME_BUS]],
							  p_mw=row[COL_NAME_P],
							  vm_pu=row[COL_NAME_VM],
							  sn_mva=row[COL_NAME_REF_POWER],
							  name=row[COL_NAME_NAME],
							  max_q_mvar=row[COL_NAME_MAX_Q],
							  min_q_mvar=row[COL_NAME_MIN_Q],
							  min_p_mw=row[COL_NAME_MIN_P],
							  max_p_mw=row[COL_NAME_MAX_P],
							  min_vm_pu=row[COL_NAME_MIN_VOLTAGE],
							  max_vm_pu=row[COL_NAME_MAX_VOLTAGE],
							  scaling=row[COL_NAME_SCALING],
							  type=row[COL_NAME_TYPE],
							  lack=row[COL_NAME_LACK],
							  controllable=row[COL_NAME_CONTROLLABLE],
							  vn_kv=row[COL_NAME_VOLTAGE],
							  xdss_pu=row[COL_NAME_XDSS],
							  rdss_ohm=row[COL_NAME_RDSS],
							  cos_phi=row[COL_NAME_COS_PHI],
							  pg_percent=row[COL_NAME_PG],
							  power_station_trafo=row[COL_NAME_PS_TRAFO],
							  in_service=row[COL_NAME_SERVICE],
							  slack_weight=row[COL_NAME_SLACK_WEIGHT],
							  slack=row[COL_NAME_SLACK]
							  )
			gen_ids[row[COL_NAME_NAME]] = pp.create_gen(
				**{key: value for key, value in kwargs_gen.items() if value is not None})

		# Creating the switch elements
		df_switch = df_switch.replace({np.nan: None})
		switches_ids = {}
		for index, row in df_switch.iterrows():
			if row[COL_NAME_ET] == 'b':
				element = bus_ids[row[COL_NAME_ELEMENT]]
			elif row[COL_NAME_ET] == 'l':
				element = line_ids[row[COL_NAME_ELEMENT]]
			elif row[COL_NAME_ET] in ['t', 't3']:
				element = tr_ids[row[COL_NAME_ELEMENT]]
			kwargs_gen = dict(net=network,
							  name=row[COL_NAME_NAME],
							  bus=bus_ids[row[COL_NAME_BUS]],
							  element=element,
							  et=row[COL_NAME_ET],
							  closed=row[COL_NAME_CLOSED],
							  in_ka=row[COL_NAME_IN_KA],
							  type=row[COL_NAME_TYPE]
							  # in_service=row[COL_NAME_SERVICE]
							  )
			switches_ids[row[COL_NAME_NAME]] = pp.create_switch(
				**{key: value for key, value in kwargs_gen.items() if value is not None})

		# Creating the cost function
		df_cost = df_cost.replace({np.nan: None})

		for index, row in df_cost.iterrows():
			if row[COL_NAME_TYPE] == 'load':
				element = load_ids[row[COL_NAME_NAME]]
			elif row[COL_NAME_TYPE] == 'ext_grid':
				element = ex_gen_ids[row[COL_NAME_NAME]]
			elif row[COL_NAME_TYPE] == 'gen':
				element = gen_ids[row[COL_NAME_NAME]]
			kwargs_cost = dict(net=network,
							   element=element,
							   et=row[COL_NAME_TYPE],
							   cp1_eur_per_mw=row[COL_NAME_CP1],
							   cp0_eur=row[COL_NAME_CP0],
							   cq1_eur_per_mvar=row[COL_NAME_CQ1],
							   cq0_eur=row[COL_NAME_CQ0],
							   cp2_eur_per_mw2=row[COL_NAME_CP2],
							   cq2_eur_per_mvar2=row[COL_NAME_CQ2])
			pp.create_poly_cost(
				**{key: value for key, value in kwargs_cost.items() if value is not None})
		return network

	def get_failure_candidates(self):
		candidates = {}
		for x in [x for x in self.__dict__.values() if isinstance(x, dict)]:
			candidates.update({y.id:y for y in x.values() if hasattr(y, 'failureProb') and y.failureProb != None and y.i_montecarlo != True})
		priorities = [x.priority if x.priority  != None else float('inf') for x in candidates.values()] # infinity for None priorities
		return {x:candidates[x] for _,x  in sorted(zip(priorities, candidates))}

	def get_closest_available_crews(self, availableCrew, powerElements):
		aux = len(powerElements)
		return powerElements[0:aux], availableCrew[0:aux]

	def get_crews_traveling_time(self, crew, powerElements):
		# time = np.random.randint(10)
		time = 0
		return [time]*len(crew)

	def get_reparing_time(self, powerElementsID, powerElements):
		# TODO: correct formula for repairing time
		return [powerElements[x].normalTTR for x in powerElementsID]

	def calculate_outages_schedule(self, simulationTime, hazardTime):
		'''
		outagesSchedule = 1 if powerElement is available
		'''
		failureCandidates = self.get_failure_candidates()

		failureProbability = np.array([x.failureProb for x in failureCandidates.values()])

		randomNumber = np.random.rand(len(failureProbability))
		failure = np.where((randomNumber <= failureProbability), 
							np.random.randint(hazardTime.start, [hazardTime.stop]*len(failureCandidates)),
							simulationTime.stop)

		crewSchedule = pd.DataFrame([[1]*len(self.crews)]*simulationTime.maxduration,
									columns=self.crews.keys(), index=simulationTime.interval)

		outagesSchedule = pd.DataFrame([[STATUS['on']]*len(failureCandidates)] *
									   simulationTime.maxduration, columns=failureCandidates.keys(), index=simulationTime.interval)

		for index, column in zip(failure, outagesSchedule):
			outagesSchedule[column].loc[index:] = STATUS['off']

		for index, row in outagesSchedule.iterrows():
			failureElements = row.index[row == 0].tolist()
			availableCrews = crewSchedule.loc[index][crewSchedule.loc[index] == 1].index.tolist()
			elementsToRepair, repairingCrews = self.get_closest_available_crews(availableCrews, failureElements)
			crewsTravelingTime = self.get_crews_traveling_time(repairingCrews, elementsToRepair)
			repairingTime = self.get_reparing_time(elementsToRepair, failureCandidates)
			for t_0, t_1, e, c in zip(crewsTravelingTime, repairingTime, elementsToRepair, repairingCrews):
				outagesSchedule.loc[index+1:index + t_0, e] = STATUS['waiting']
				outagesSchedule.loc[index+t_0+1:index + t_0 + t_1, e] = STATUS['reparing']
				# Following line can be removed if outagesSchedule is set to 1 on at failure time
				outagesSchedule.loc[index+t_0+t_1+1:, e] = STATUS['on']
				crewSchedule.loc[index+1:index+t_0+t_1, c] = e

			if not outagesSchedule.loc[index+1:].isin([STATUS['on']]).any().any():
				# print(outagesSchedule.join(crewSchedule))
				self.outagesSchedule = outagesSchedule
				self.crewSchedule = crewSchedule
				return

		self.outagesSchedule = outagesSchedule
		self.crewSchedule = crewSchedule

	def get_switch_candidates(self):
		return {k:v for k,v in self.switches.items() if v.associated_elements != None and v.i_montecarlo != True}

	def calculate_switches_schedule(self, simulationTime):
		"""
		Switches will negate 'closed_' state iif at least one of its disconnecting_element is in outagesSchedule and its value is zero. This is done by a XNOR between closed_ values and previous condition.
		switchesSchedule = 1 iif switch is closed
		OBS: If a powerElement is excluded from self.outagesSchedule (e.g. i_montecarlo = True) then it will play no role on the determination of its associated switches' status.
		"""
		switchesSchedule = pd.DataFrame.from_dict(
			{k: [v.closed_]*simulationTime.maxduration for k, v in self.get_switch_candidates().items()}, dtype='int')  # forcing int is important!
		
		switchesSchedule.index = simulationTime.interval
		for switch_id in switchesSchedule:
			filter = self.outagesSchedule.columns.intersection(
				self.switches[switch_id].associated_elements)
			switchesSchedule[switch_id] = (switchesSchedule[switch_id] == (self.outagesSchedule[filter] > 0).all(axis=1))*1 #XNOR
		self.switchesSchedule = switchesSchedule

	def get_powerelement(self, id):
		# It assumes ids are unique!
		return next((x[id] for x in [self.loads, self.generators, self.transformers, self.lines, self.switches, self.nodes] if id in x), None)

	def propagate_schedules_to_network_elements(self):
		"""
		Assigns 'in_service' field based on outagesSchedule for powerElements
		Assigns 'closed' field based on switchesSchedule for switches
		"""
		df_in_service = self.outagesSchedule > 0
		df_closed = self.switchesSchedule > 0
		iterator = [['in_service', df_in_service], ['closed', df_closed]]
		for field, y in iterator:
			for elementId in y:
				element = self.get_powerelement(elementId)
				setattr(element, field, y[elementId])
				self.update_montecarlo_variables(standard_dict(content=element, type=type(
					element).__name__, id=elementId, field=field))  # elements are pointers

	def update_montecarlo_variables(self, standard_dict):
		element = find_element_in_standard_dict_list(
			standard_dict['id'], self.mcVariables)
		if not element or element['field'] != standard_dict['field']:
			self.mcVariables.append(standard_dict)

	def update_grid(self, montecarlo_database, debug=None):
		for type, id, field in montecarlo_database.columns.dropna():  # useful for empty database
			if id == debug and field == 'in_service':
				print(f'before update_grid: {debug} is {self.get_powerelement(id).in_service}')
			setattr(self.get_powerelement(id), field,
					montecarlo_database[type, id, field])
			if id == debug and field == 'in_service':
				print(f'after update_grid {debug} is {self.get_powerelement(id).in_service}')

	def build_montecarlo_database(self, time):
		return build_database(self.mcVariables).loc[time.start: time.stop-1]

	def build_timeseries_database(self, time):
		""" 
		From the list [loads, generators, transformers, lines, switches] it creates a database with all the fields corresponding to timeseries
		TODO: Replace list by a function that will recognise powerElement-type instances
		"""
		return build_database(get_datatype_elements([self.loads, self.generators, self.transformers, self.lines, self.switches, self.nodes], TIMESERIES_CLASS)).loc[time.start: time.stop-1]

	def calculate_metrics(self):
		# DEPRECATED
		self.metrics = []

		def elements_in_service():
			return self.outagesSchedule[self.outagesSchedule > 0].sum(axis=1)

		def crews_in_service():
			return (self.crewSchedule != 1).sum(axis=1)

		self.metrics.append(Metric('Network', 'crews_in_service', crews_in_service()))
		self.metrics.append(Metric('Network', 'elements_in_service', elements_in_service()))
		self.metrics.append(Metric('Network', 'total_elements_in_service', elements_in_service().sum(), subfield='subfield_1', unit='unit_1'))

	def build_metrics_database(self):
		# DEPRECATED
		# TODO: make it recurrent
		out = []
		for metric in self.metrics:
			keys, values = zip(*[(key, value) for key, value in metric.__dict__.items() if key != 'value'])
			if isinstance(metric.value, pd.DataFrame) or isinstance(metric.value, pd.Series):
				value = metric.value.values
				index = metric.value.index
			else:
				value = metric.value
				index = [0]
			df = pd.DataFrame(value, columns=pd.MultiIndex.from_tuples(
				[values], names=keys), index=index)
			out.append(df)
		return pd.concat(out, axis=1)

	def run(self, time, debug=None, **kwargs):
		# TODO: to include an argument to choose which elements will not be considered in the timeseries simulation. For instance, running a simulation with/without switches
		return self.calculationEngine.run(self.build_timeseries_database(time), debug=debug, **kwargs)

	def calc_stratas(self, data, ref_rp, xmin, xmax, cv=0.1, maxStrata=10):
		if maxStrata == 1:
			return pd.DataFrame({'Allocation': [1], 'Lower_X1': [xmin], 'Upper_X1': [xmax]})
		else:
			stratify = importr('SamplingStrata')
			base = importr('base')
			pandas2ri.activate()

			ids = list(range(len(data)))
			domain = [1]*len(data)

			Y = ["id"]
			y = [ids]
			cv_ = ["DOM1"]
			CV_ = ["DOM"]

			fc_rp_list = []
			for _, value in self.powerElements.items():
				if value.fragilityCurve != None and value.return_period != None:
					temp = [value.fragilityCurve, value.return_period]
					if temp not in fc_rp_list:
						fc_rp_list.append(temp)

			n = 1
			for fc_rp in fc_rp_list:
				Y.append(fc_rp[0]+'_'+fc_rp[1])
				y.append(self.fragilityCurves[fc_rp[0]].projected_fc(self.returnPeriods[fc_rp[1]], ref_rp, data))
				cv_.append(cv)
				CV_.append(str("CV" + str(n)))
				n += 1

			Y.extend(["x", "domain"])
			y.extend([data, domain])
			cv_.append(1)
			CV_.append("domainvalue")

			df = pd.DataFrame(np.transpose(np.array(y)), columns=Y)

			frame = stratify.buildFrameDF(df=robjects.conversion.py2rpy(df),
											id = "id", X = ["x"], Y = Y[1:-2],
											domainvalue = "domain")

			df_cv = base.as_data_frame(rlc.TaggedList(cv_, tags=tuple(CV_)))

			kmean = stratify.KmeansSolution2(frame=frame,
												errors=df_cv,
												maxclusters = maxStrata,
												showPlot = False)
			try:
				if maxStrata == 1:
					nstrat = 1
				else:
					nstrat = np.unique(robjects.conversion.rpy2py(kmean)["suggestions"].values).size
				sugg = stratify.prepareSuggestion(kmean=kmean, frame=frame, nstrat=nstrat)
				solution = stratify.optimStrata(method= "continuous", errors = df_cv,
                                    framesamp = frame, iter = 50,
                                    pops = 20, nStrata = nstrat,
									suggestions = sugg,
                                    showPlot = False,
                                    parallel=True)
			except:
				warnings.warn(
				    f'Cannot find optimal number of stratas... Please try with a different reference return period. Continuing with 5 stratas')
				solution = stratify.optimStrata(method= "continuous", errors = df_cv,
                                    framesamp = frame, iter = 50,
											pops = 20, nStrata = 5,
                                    showPlot = False,
                                    parallel=True)
			strataStructure = stratify.summaryStrata(robjects.conversion.rpy2py(solution)[2],
                                            robjects.conversion.rpy2py(solution)[1],
                            progress=False)
			return (robjects.conversion.rpy2py(strataStructure))

class MonteCarloVariable:
	def __init__(self, element, id, field):
		self.element = element
		self.id = id
		self.field = field

class Metric:
	# TODO: delete
	def __init__(self, network_element, field, value, subfield=None, unit=None):
		self.network_element = network_element
		self.field = field
		self.value = value
		self.subfield = subfield
		self.unit = unit

class GeoData:
	'''
	TODO: @TIM add description
	Add description of GeoData class
	'''
	def __init__(self, latitude, longitude):
		self.latitude = latitude
		self.longitude = longitude


class PowerElement:
	'''
	TODO: @TIM add description
	Add description of PowerElement class here

	:attribute fragilityCurve: string, the type of the fragility curve
	'''
	def __init__(self, **kwargs):
		self.id = None
		self.node = None
		self.failureProb = None
		self.fragilityCurve = None
		self.normalTTR = None
		self.in_service = None
		self.i_montecarlo = None
		self.priority = None
		self.return_period = None

		for key, value in kwargs.items():
			if hasattr(self, key):
				setattr(self, key, value)
			else:
				warnings.warn(f'Input parameter "{key}" is not a valid attribute of the "{self.__class__.__name__}" class.')
		if self.id is None:
			self.id = utils.get_GLOBAL_ID()
			warnings.warn(f'No "id" defined as input for {self}. The folloging identifier was assigned: {self.id}.')
		if self.in_service == False:
			if self.i_montecarlo == False:
				warnings.warn(f'Forcing "ignore montecarlo" to "True" for {self.id}.')
			self.i_montecarlo = True

	def initiate_failure_parameters(self,
									fragilityCurve=None,
									kf=None,
									resilienceFull=None,
									failureProb=0,
									inReparation=False,
									normalTTR=None,
									distTTR=0,
									in_service=True):

		self.fragilityCurve = fragilityCurve
		self.kf = kf
		self.resilienceFull = resilienceFull
		self.failureProb = failureProb
		self.inReparation = inReparation
		self.normalTTR = normalTTR
		self.distTTR = distTTR
		self.in_service = in_service

	def update_failure_probability(self, network, intensity=None, ref_return_period=None):
		if self.fragilityCurve == None:
			self.failureProb = None
		else:
			node = network.nodes[self.node]
			if intensity == None and network.event.intensity is not None:
				_, event_intensity = network.event.get_intensity(node.longitude, node.latitude)
				intensity = event_intensity.max()
			elif network.event.intensity is None:
				warnings.warn(f'Hazard event is not defined')

			if self.return_period != None and ref_return_period != None:
				self.failureProb = network.fragilityCurves[self.fragilityCurve].projected_fc(rp=network.returnPeriods[self.return_period],
																							ref_rp=network.returnPeriods[ref_return_period],
																							xnew=intensity)[0]
			else:
				self.failureProb = network.fragilityCurves[self.fragilityCurve].interpolate(intensity)[0]


class Bus(PowerElement):
	'''
	TODO: @TIM add description
	Class Bus: Parent Class PowerElement

	:attribute longitude: float, longitude in degrees
	:attribute latitude: float, latitude in degrees
	'''
	def __init__(self, kwargs):
		self.vn_kv = None
		self.in_service = None
		self.weatherTTR = None  # TODO: Firas's code
		self.elapsedReparationTime = None  # TODO: Firas's code
		self.longitude = None
		self.latitude = None
		self.max_vm_pu = None
		self.min_vm_pu = None
		super().__init__(**kwargs)

	def update_failure_probability(self, network, intensity=None, ref_return_period=None):		
		if self.fragilityCurve == None:
			self.failureProb = None
		else:
			if intensity == None:
				_, event_intensity = network.event.get_intensity(self.longitude, self.latitude)
				intensity = event_intensity.max()
			if self.return_period != None and ref_return_period != None:
				self.failureProb = network.fragilityCurves[self.fragilityCurve].projected_fc(rp=network.returnPeriods[self.return_period],
																							ref_rp=network.returnPeriods[ref_return_period],
																							xnew=intensity)[0]
			else:
				self.failureProb = network.fragilityCurves[self.fragilityCurve].interpolate(intensity)[0]

class Switch(PowerElement):
	# TODO: @TIM add description
	def __init__(self, kwargs):
		# element type: “l” = switch between bus and line, “t” = switch between bus and transformer, “t3” = switch between bus and transformer3w, “b” = switch between two buses'
		self.et = None
		self.element = None
		self.closed = None
		self.in_ka = None
		self.type = None
		self.associated_elements = None
		super().__init__(**kwargs)
		self.closed_ = self.closed
		if self.associated_elements:
			self.associated_elements = [
				x.strip() for x in self.associated_elements.split(',')]

class Generator(PowerElement):
	'''
	TODO: @TIM add description
	Add description of Generator class here
	'''
	def __init__(self, kwargs):
		self.p_mw = None
		self.q_mvar = None
		self.vm_pu = None
		self.controllable = None
		self.max_p_mw = None
		self.min_p_mw = None
		self.max_q_mvar = None
		self.min_q_mvar = None
		self.slack = None
		self.weatherTTR = None  # TODO: Firas's code
		self.elapsedReparationTime = None  # TODO: Firas's code
		super().__init__(**kwargs)


class Load(PowerElement):
	'''
	TODO: @TIM add description
	Add description of Generator class here
	'''
	def __init__(self, kwargs):
		self.p_mw = None
		self.q_mvar = None
		self.controllable = None
		self.max_p_mw = None
		self.min_p_mw = None
		self.max_q_mvar = None
		self.min_q_mvar = None
		super().__init__(**kwargs)
		# self.node_id = node_id


class Transformer(PowerElement):
	'''
	TODO: @TIM add description
	Add description of Transformer class here
	'''
	def __init__(self, kwargs):
		self.node_p = None
		self.node_s = None
		self.type = None
		self.vn_hv_kv = None
		self.vn_lv_kv = None
		self.sn_mva = None
		self.vk_percent = None
		self.vkr_percent = None
		self.pfe_kw = None
		self.i0_percent = None
		self.shift_degree = None
		self.tap_side = None
		self.tap_neutral = None
		self.tap_min = None
		self.tap_max = None
		self.tap_step_percent = None
		self.tap_step_degree = None
		self.tap_phase_shifter = None
		self.max_loading_percent = None
		self.weatherTTR = None  # TODO: Firas's code
		self.elapsedReparationTime = None  # TODO: Firas's code
		super().__init__(**kwargs)

	def update_failure_probability(self, network, intensity=None, ref_return_period=None):
		if self.fragilityCurve == None:
			self.failureProb = None
		else:
			node = network.nodes[self.node_p]
			if intensity == None:
				_, event_intensity = network.event.get_intensity(node.longitude, node.latitude)
				intensity = event_intensity.max()
			if self.return_period != None and ref_return_period != None:
				self.failureProb = network.fragilityCurves[self.fragilityCurve].projected_fc(rp=network.returnPeriods[self.return_period],
																							ref_rp=network.returnPeriods[ref_return_period],
																							xnew=intensity)[0]
			else:
				self.failureProb = network.fragilityCurves[self.fragilityCurve].interpolate(intensity)[0]

class Line(PowerElement):
	'''
	TODO: @TIM add description
	Add description of Line class here
	'''
	def __init__(self, kwargs):
		self.from_bus = None
		self.to_bus = None
		self.length_km = None
		self.r_ohm_per_km = None
		self.x_ohm_per_km = None
		self.c_nf_per_km = None
		self.r0_ohm_per_km = None
		self.x0_ohm_per_km = None
		self.c0_nf_per_km = None
		self.max_i_ka = None
		self.g_us_per_km = None
		self.g0_us_per_km = None
		self.type = None
		self.max_loading_percent = None
		self.span = None  # TODO: Firas's code
		self.lineSpan = None  # TODO: Firas's code
		self.towers = None  # TODO: Firas's code
		super().__init__(**kwargs)

	def update_failure_probability(self, network, intensity=None, ref_return_period=None):
		if self.fragilityCurve == None:
			self.failureProb = None
		else:
			node1 = network.nodes[self.from_bus]
			node2 = network.nodes[self.to_bus]
			nb_segments = math.ceil(self.length_km/self.lineSpan)
			probFailure = []
			for i_segment in range(nb_segments+1):
				lon = node1.longitude + i_segment*(node2.longitude-node1.longitude)/nb_segments
				lat = node1.latitude + i_segment*(node2.latitude-node1.latitude)/nb_segments
				if intensity == None:
					_, event_intensity = network.event.get_intensity(lon, lat)
					intensity = event_intensity.max()
				if self.return_period != None and ref_return_period != None:
					probFailure.append(network.fragilityCurves[self.fragilityCurve].projected_fc(rp=network.returnPeriods[self.return_period],
																								ref_rp=network.returnPeriods[ref_return_period],
																								xnew=intensity))
				else:
					probFailure.append(network.fragilityCurves[self.fragilityCurve].interpolate(intensity))		
			self.failureProb = 1-np.prod(1-np.array(probFailure))

class Crew:
	'''
	TODO: @TIM add description
	Add description of Crew class here
	'''
	def __init__(self, kwargs):
		# self.available = number
		self.id = None
		self.geodata = None
		for key, value in kwargs.items():
			if hasattr(self, key):
				setattr(self, key, value)
			else:
				warnings.warn(f'Input parameter "{key}" unknown in {kwargs} .')
		if self.geodata is None:
			self.geodata = GeoData(0, 0)