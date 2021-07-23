import os
import numpy as np
import tensorflow as tf
import subprocess as sp
import csv

#####################
# MODEL RUN HELPERS #
#####################

'''
Select gpu and allocate a maximum fraction of memory on it
Args:
	- gpu: PCI bus number of gpu to use (str)
	- frac: Fraction of memory to allocate (float)
Returns:
	- config: Don't know what this is, give to InteractiveSession
'''
def set_gpu(gpu, frac):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu
	gpu_options = GPUOptions(per_process_gpu_memory_fraction=frac)
	config = ConfigProto(gpu_options=gpu_options)
	return config

'''
Returns total memory capacity of all gpus available (list)
'''
def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
  memory_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_values = [int(x.split()[0]) for i, x in enumerate(memory_info)]
  return memory_values

'''
Selects gpu and fraction of its memory to use
Args:
	- gpu_no: Gpu number identifier (str)
	- memory_fraction: Fraction of total memory from selected gpu to use (float (0,1))
	- gpu_memories: List of total memory of all gpus available (list)
Returns:
	- gpus: List of physical gpu devices as tensorflow talks about them (list)
	- logical_gpus: List of created virtual gpus created to allocate space (list)
Note: In current implementation, there should always be 1 and 1 physical and logical gpu
'''
def configure_gpu(gpu_no, memory_fraction, gpu_memories):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
  memory_limit = gpu_memories[int(gpu_no)] * memory_fraction
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(gpus[0],\
     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
  logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  return gpus, logical_gpus

'''
Make directories to save model data if they don't exist yet
Args:
	- cond_dir: Condition-named directory - under "ModelData" (str)
	- run_dir: Run number-named directory - under cond_dir (str)
'''
def make_dirs(cond_dir, run_dir):
	try:
		os.mkdir(cond_dir)
	except OSError:
		pass
	try:
		os.mkdir(run_dir)
	except OSError:
		pass

################
# LOADING DATA #
################

'''
Load trainable variables from file
Args:
	- run_dir: Run number-named directory - Model->cond_dir->run_dir->[files] (str)
Returns:
	- trainable_vars: List of trainable vars as numpy arrays
'''
def load_trainable(run_dir):
	trainable_vars = []
	for file in sorted(os.listdir(run_dir)): # Impose alphanumeric order
		if ("w" in file) | ("w_out" in file) | ("taus_gaus" in file) | ("bias" in file):
			var = np.loadtxt(run_dir + "/" + file, delimiter=",")
			trainable_vars.append(var)
	return trainable_vars
'''
Get hyperparams from a run into a dictionary
Args:
	- run_dir: Run number-named directory - ModelData->cond_dir->run_dir->[files] (str)
Returns:
	- hyperparams: Dictionary of hyperparameters
'''
def get_hyperparams(run_dir):
	hyperparams = {}
	with open(run_dir + "/hyperparams.csv" , "r") as f:
		reader = csv.reader(f)
		for row in reader:
			hyperparams[row[0]] = row[1]
	return hyperparams

'''
Get model data from a run into a dictionary
Args:
	- run_dir: Run number-named directory - ModelData->cond_dir->run_dir->[files] (str)
Returns:
	- data: Dictionary of model data
'''
def get_model_data(run_dir):
	data = {}
	for file in os.listdir(run_dir):
		if (".csv" in file) & ("hyperparams" not in file):
			data[file[:-4]] = np.loadtxt(run_dir + f"/{file}", delimiter=",")
	return data

# Get operational synaptic time constants
def get_taus_sig(taus_gaus, taus):
	# Prepare synaptic time constants
	taus_sig = sigmoid(taus_gaus) * (taus[1] - taus[0]) + taus[0]
	return taus_sig

###################
# ACTIVATION FCNS #
###################

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
	return np.exp(-x) / ((1 + np.exp(-x))**2)

def tanh(x):
	return np.tanh(x)

# def tanh(x):
# 	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_prime(x):
	return 1 / np.square(np.cosh(x))

def linear(x):
	return x

#######################
# FIXED POINT FINDING #
#######################

'''
Computes the jacobian matrix of the system at point x
Args:
	- x: neural state (n,)
	- u: input to network (n_inputs, 1)
	- net: object of class "rnn" from Model.py
	- activation_fcn: Specify which used in net ("tanh" or "sigmoid")
Returns:
	- q value: (float)
'''
def net_jacobian(x, u, net, activation_fcn):
	x = x.reshape(1,-1)
	ww = net.get_ww()
	taus = net.get_taus_sig().reshape(-1,) # Broadcasts with col vector x as row vector would but must be 1D like this to work with diag

	if activation_fcn == "tanh":
		activation_prime = tanh_prime
	elif activation_fcn == "sigmoid":
		activation_prime = sigmoid_prime

	jacobian = np.multiply(ww, np.multiply(1 / taus, activation_prime(x))) - np.diag(1 / taus)
	return jacobian

'''
Returns q value of a given neural state, x
Args:
	- x: neural state (n,)
	- u: input to network (n_inputs, 1)
	- net: object of class "rnn" from Model.py
	- activation_fcn: Specify which used in net ("tanh" or "sigmoid")
Returns:
	- q value: (float)
'''
def q(x, u, net, activation_fcn):
	x = x.reshape(-1,1) # Provided as numpy 1D array, now "2D" column vector
	f_x = net.f_x(x, u)
	q = 0.5 * np.sum(np.square(f_x))
	return q

'''
Gradient of q
Args:
	- x: neural state (n,)
	- u: input to network (n_inputs, 1)
	- net: object of class "rnn" from Model.py
	- activation_fcn: Specify which used in net ("tanh" or "sigmoid")
Returns:
	- grad_q: Gradient of q (n,) 
'''
def grad_q(x, u, net, activation_fcn):
	x = x.reshape(-1,1) # Provided as numpy 1D array, now "2D" column vector
	jacobian = net_jacobian(x, u, net, activation_fcn)
	f = net.f_x(x, u)
	grad_q = np.matmul(jacobian.T, f).reshape(-1,)
	return grad_q

'''
Hessian of q
Args:
	- x: neural state (n,)
	- u: input to network (n_inputs, 1)
	- net: object of class "rnn" from Model.py
	- activation_fcn: Specify which used in net ("tanh" or "sigmoid")
Returns:
	- hess_q: Hessian matrix (n,n)
'''
def hess_q(x, u, net, activation_fcn):
	jacobian = net_jacobian(x, u, net, activation_fcn)
	hess_q = np.matmul(jacobian.T, jacobian)
	return hess_q

# End