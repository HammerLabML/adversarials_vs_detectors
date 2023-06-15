import yaml

def read():
	with open("bisection_5d5h5m_1.yaml", "r") as stream:
		try:
			obj = yaml.safe_load(stream)
			return obj
		except yaml.YAMLError as exc:
			print(exc)

def write():
	my_dict = {'200': ['10', '20'], '400': [], '300': ['10']}
	with open('test.yaml', 'w') as fp:
		yaml.dump(my_dict, fp, sort_keys=False, default_flow_style=None)

if __name__=='__main__':
	write()
