import csv
import sys
path = '/cnl/data/spate/Rnn'
if path not in sys.path:
	sys.path.append(path)
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import Tasks as tasks
import Helpers as h

class rnn:
	def __init__(self, n_units, n_inputs, n_outputs, p_con, dale, p_inh, g, taus, activation_fcn, seed=None, restored_vars=None):
		# User inputted
		self.n_units = n_units
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.p_con = p_con # Probability of connection
		self.dale = dale
		self.p_inh = p_inh # Prob inhibitory
		self.g = g # Recurrent connection gain
		self.taus = taus # Synaptic timescales
		if activation_fcn == "tanh":
			self.activation_fcn_tf = tf.tanh
			self.activation_fcn_np = h.tanh
		elif activation_fcn == "sigmoid":
			self.activation_fcn_tf = tf.sigmoid
			self.activation_fcn_np = h.sigmoid
		elif activation_fcn == "linear":
			self.activation_fcn_tf = tf.keras.activations.linear
			self.activation_fcn_np = h.linear
		self.seed = seed # Random seed for parameter intialization
		self.restored_vars = restored_vars # w, w_out, taus_gaus, out_bias

		w_init, w_out_init, taus_gaus_init, out_bias_init, bias_init = self.init_vars()

		# Internal
		self.delta_t = 1 # Sampling rate
		self.w_in = tf.Variable(tf.random.normal(shape=(self.n_units, self.n_inputs), seed=self.seed), name="w_in") # Input weights
		self.mask = tf.Variable(self.init_mask(), name="mask", shape=(self.n_units, self.n_units))
		self.taus_gaus = tf.Variable(taus_gaus_init, name="taus_gaus", shape=(self.n_units, 1))
		
		# Trainable
		self.out_bias = tf.Variable(out_bias_init, name="out_bias", shape=(self.n_outputs, 1)) # Output bias
		self.bias = tf.Variable(bias_init, name='bias', shape=(self.n_units, 1)) # State bias
		self.w = tf.Variable(w_init, name="w", shape=(self.n_units, self.n_units)) # Connectivity matrix
		self.w_out = tf.Variable(w_out_init, name="w_out", shape=(self.n_outputs, self.n_units))
		self.trainable_vars = [self.bias, self.out_bias, self.w, self.w_out]

	'''
	Simulate network dynamics for one trial
	'''
	@tf.function
	def simulate(self, u, trial_len, batch_size=1):
		u = tf.reshape(tf.cast(u, tf.float32), shape=(self.n_inputs, trial_len, batch_size)) # Input
		
		# Use these to store stuff; list appending messes up tf.function
		output = tf.TensorArray(tf.float32, trial_len)
		syncur = tf.TensorArray(tf.float32, trial_len)
		
		x = tf.random.normal(shape=(self.n_units, batch_size)) / 100 # Initial state of syncur
		
		# Prepare recurrent weight matrix
		if self.dale:
			w = tf.abs(self.w)
		else:
			w = self.w
		ww = tf.matmul(w, self.mask)

		# Prepare synaptic time constants
		taus_sig = tf.sigmoid(self.taus_gaus) * (self.taus[1] - self.taus[0]) + self.taus[0]
		taus_sig = tf.tile(taus_sig, [1, batch_size])
		# taus_sig = tf.repeat(taus_sig, batch_size, axis=1)

		bias = tf.tile(self.bias, [1, batch_size]) # Repeat state bias over batches
		
		# Run network through a trial
		for t in range(trial_len):
			r = self.activation_fcn_tf(x)
			next_x = tf.multiply((1 - self.delta_t / taus_sig), x)\
				+ tf.multiply((self.delta_t / taus_sig), ((tf.matmul(ww, r))\
				+ tf.matmul(self.w_in, u[:,t,:])))\
				+ bias\
				+ tf.random.normal(shape=(self.n_units, batch_size)) / 100
			next_o = tf.matmul(self.w_out, self.activation_fcn_tf(next_x)) + self.out_bias
			x = next_x
			output = output.write(t, next_o)
			syncur = syncur.write(t, x)
		
		# Prepare to return stored data
		output = tf.transpose(output.stack(), perm=[1,0,2]) # Transpose to [n_outputs x trial_len x batch_size]
		syncur = tf.transpose(syncur.stack(), perm=[1,0,2]) # Transpose to [n_units x trial_len x batch_size]
		return output, syncur

	'''
	Train network one step
	'''
	@tf.function
	def train(self, u, target, trial_len, optimizer, batch_size):
		target = tf.reshape(tf.cast(target, tf.float32), shape=(self.n_outputs, trial_len, batch_size))
		with tf.GradientTape() as tape: # Gradient tape keeps track of graph during eager execution
			tape.watch(self.trainable_vars) # Tell gradient tape vars to keep track of
			output, _ = self.simulate(u, trial_len, batch_size)
			loss = self.loss_fcn(output, target)
			gradients = tape.gradient(loss, self.trainable_vars)
			optimizer.apply_gradients(zip(gradients, self.trainable_vars))
		return loss

	'''
	Initialize mask
	'''
	def init_mask(self):
		mask = np.eye(self.n_units)
		if self.dale:
			np.random.seed(self.seed)
			self.inh = np.random.rand(self.n_units) < self.p_inh
			self.exc = ~self.inh
			mask[self.inh, self.inh] = -1
			np.random.seed(None)
		return mask.astype(np.float32)

	'''
	Initialize trainable variables to restored values or seeded random values
	'''
	def init_vars(self):
		np.random.seed(self.seed)

		if self.restored_vars != None:
			bias_init, out_bias_init, w_init, w_out_init = self.restored_vars # Restore in alphabetical order. Imposed in helper fcn "load_trainable"
			bias_init, out_bias_init, w_init, w_out_init = bias_init.reshape(self.n_units, 1), out_bias_init.reshape(self.n_outputs, 1), w_init.reshape(self.n_units, self.n_units), w_out_init.reshape(self.n_outputs, self.n_units)
		else:
			bias_init = np.zeros(shape=(self.n_units, 1))
			out_bias_init = np.zeros(shape=(self.n_outputs, 1))
			w_out_init = np.random.normal(size=(self.n_outputs, self.n_units)) / 100
			w_init = np.zeros((self.n_units, self.n_units))
			idx = np.random.rand(self.n_units, self.n_units) < self.p_con
			w_init[idx] = np.random.normal(size=idx.sum()) * (self.g / np.sqrt(self.n_units * self.p_con))
		
		taus_gaus_init = np.random.normal(size=(self.n_units, 1))	
		np.random.seed(None)
		return w_init.astype(np.float32), w_out_init.astype(np.float32), taus_gaus_init.astype(np.float32), out_bias_init.astype(np.float32), bias_init.astype(np.float32)
		
	'''
	Loss function
	'''
	@tf.function
	def loss_fcn(self, output, target):
		loss = tf.reduce_mean(tf.square(tf.subtract(output, target))) # Mean squared error
		return loss
	

	# Need fix, add state bias
	'''
	Return delta of one step of the rnn
	'''
	# def f_x(self, x, u):
	# 	x = x.reshape(-1,1)
	# 	u = u.reshape(-1,1)
	# 	# Prepare recurrent weight matrix
	# 	if self.dale:
	# 		w = abs(self.w.numpy())
	# 	else:
	# 		w = self.w.numpy()
	# 	ww = np.matmul(w, self.mask.numpy())

	# 	# Prepare synaptic time constants
	# 	taus_sig = h.sigmoid(self.taus_gaus.numpy()) * (self.taus[1] - self.taus[0]) + self.taus[0]
		
	# 	# Run one step 
	# 	r = self.activation_fcn_np(x)
	# 	next_x = np.multiply((1 - self.delta_t / taus_sig), x)\
	# 			+ np.multiply((self.delta_t / taus_sig), ((np.matmul(ww, r))\
	# 			+ np.matmul(self.w_in.numpy(), u)))\
	# 			# + np.random.randn(self.n_units, batch_size) / 10
	# 	delta_x = next_x - x # Get change in rnn state
	# 	return delta_x
	
	'''
	Getters
	'''
	# Get trainable variables
	def get_trainable(self):
		return self.trainable_vars

	# Get operational recurrent connectivity matrix
	def get_ww(self):
		# Prepare recurrent weight matrix
		if self.dale:
			w = abs(self.w.numpy())
		else:
			w = self.w.numpy()
		ww = np.matmul(w, self.mask.numpy())
		return ww
	
	# Get operational synaptic time constants
	def get_taus_sig(self):
		# Prepare synaptic time constants
		taus_sig = h.sigmoid(self.taus_gaus.numpy()) * (self.taus[1] - self.taus[0]) + self.taus[0]
		return taus_sig
	
	# Get output of state x
	def get_output(self, x):
		x = x.reshape(-1,1) # Reshape to column vector
		o = np.matmul(self.w_out.numpy(), self.activation_fcn_np(x)) + self.out_bias.numpy()
		return o