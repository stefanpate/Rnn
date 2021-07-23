'''
Port changes made in Model, FlipFlop backwards to this file
'''
# import csv
# import sys
# path = '/cnl/data/spate/Rnn'
# if path not in sys.path:
# 	sys.path.append(path)
# import numpy as np
# import os
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import Tasks as tasks
# import Helpers as h
# from tensorflow.compat.v1 import InteractiveSession

# #########
# # MODEL #
# #########
# class rnn:
# 	def __init__(self, n_units, n_inputs, n_outputs, p_con, p_inh, g, taus, activation_fcn, seed=None, restored_vars=None):
# 		# User inputted
# 		self.n_units = n_units
# 		self.n_inputs = n_inputs
# 		self.n_outputs = n_outputs
# 		self.p_con = p_con # Probability of connection
# 		self.p_inh = p_inh # Prob inhibitory
# 		self.g = g # Recurrent connection gain
# 		self.taus = taus # Synaptic timescales
# 		self.activation_fcn = activation_fcn
# 		self.seed = seed # Random seed for parameter intialization
# 		self.restored_vars = restored_vars # w, w_out, taus_gaus, bias

# 		# Internal
# 		self.delta_t = 1 # Sampling rate
# 		self.w_in = tf.Variable(tf.random.normal(shape=(self.n_units, self.n_inputs), seed=self.seed), trainable=False, name="w_in") # Input weights
# 		self.mask = tf.Variable(self.init_mask(), trainable=False, name="mask", shape=(self.n_units, self.n_units))
# 		self.x_start = tf.Variable(tf.random.normal(shape=(self.n_units, 1)), trainable=False, name="x_start")
		
# 		# Trainable
# 		w_init, w_out_init, taus_gaus_init, bias_init = self.init_trainable_vars()
# 		self.w = tf.Variable(w_init, trainable=True, name="w", shape=(self.n_units, self.n_units)) # Connectivity matrix
# 		self.w_out = tf.Variable(w_out_init, trainable=True, name="w_out", shape=(self.n_outputs, self.n_units))
# 		self.taus_gaus = tf.Variable(taus_gaus_init, trainable=True, name="taus_gaus", shape=(self.n_units, 1))
# 		self.bias = tf.Variable(bias_init, trainable=True, name="bias", shape=(self.n_outputs, 1))
# 		self.trainable_vars = [self.w, self.w_out, self.taus_gaus, self.bias]

# 	'''
# 	Simulate network dyanmics for one trial
# 	'''
# 	@tf.function
# 	def simulate(self, u, trial_len):
# 		u = tf.reshape(tf.cast(u, tf.float32), shape=(-1, trial_len))
# 		output = tf.TensorArray(tf.float32, trial_len) # Store model outputs here. List appending messes up tf functions
# 		syncur = tf.TensorArray(tf.float32, trial_len)
# 		for t in range(trial_len):
# 			if t == 0:
# 				x = self.x_start
# 				r = activation_fcn(x)
# 				w = tf.abs(self.w) # Didn't like when I assigned self.w here. Switched to python obj w
# 			else:
# 				x = next_x
# 				r = activation_fcn(next_x)
# 			ww = tf.matmul(w, self.mask)
# 			taus_sig = tf.sigmoid(self.taus_gaus) * (self.taus[1] - self.taus[0]) + self.taus[0]
# 			next_x = tf.multiply((1 - self.delta_t / taus_sig), x)\
# 				 + tf.multiply((self.delta_t / taus_sig), ((tf.matmul(ww, r))\
# 				 + tf.matmul(self.w_in, tf.expand_dims(u[:,t], 1))))\
# 				 + tf.random.normal(shape=(self.n_units, 1)) / 10
# 			next_o = tf.matmul(self.w_out, activation_fcn(next_x)) + self.bias
# 			output = output.write(t, next_o)
# 			syncur = syncur.write(t, x)
# 		output = tf.transpose(tf.reshape(output.concat(), shape=(trial_len, self.n_outputs)))
# 		syncur = tf.transpose(tf.reshape(syncur.concat(), shape=(trial_len, self.n_units)))
# 		return output, syncur
	
# 	'''
# 	Train network one step
# 	'''
# 	@tf.function
# 	def train(self, u, target, trial_len, optimizer):
# 		target = tf.reshape(tf.cast(target, tf.float32), shape=(self.n_outputs, trial_len))
# 		with tf.GradientTape() as tape: # Gradient tape keeps track of graph during eager execution
# 			tape.watch(self.trainable_vars) # Tell gradient tape vars to keep track of
# 			output, _ = self.simulate(u, trial_len)
# 			loss = self.loss_fcn(output, target)
# 			gradients = tape.gradient(loss, self.trainable_vars)
# 			optimizer.apply_gradients(zip(gradients, self.trainable_vars))
# 		return loss

# 	'''
# 	Initialize mask
# 	'''
# 	def init_mask(self):
# 		np.random.seed(self.seed)
# 		self.inh = np.random.rand(self.n_units) < self.p_inh
# 		self.exc = ~self.inh
# 		mask = np.eye(self.n_units)
# 		mask[self.inh, self.inh] = -1
# 		np.random.seed(None)
# 		return mask.astype(np.float32)

# 	'''
# 	Initialize trainable variables to restored values or seeded random values
# 	'''
# 	def init_trainable_vars(self):
# 		np.random.seed(self.seed)
# 		if self.restored_vars != None:
# 			bias_init, taus_gaus_init, w_init, w_out_init = self.restored_vars # Restore in alphabetical order. Imposed in helper fcn "load_trainable"
# 			bias_init, taus_gaus_init, w_init, w_out_init = bias_init.reshape(self.n_outputs, 1),\
# 				taus_gaus_init.reshape(self.n_units, 1), w_init.reshape(self.n_units, self.n_units),\
# 				w_out_init.reshape(self.n_outputs, self.n_units)
# 		else:
# 			w_init = np.zeros((self.n_units, self.n_units))
# 			idx = np.random.rand(self.n_units, self.n_units) < self.p_con
# 			w_init[idx] = np.random.normal(size=idx.sum()) * (self.g / np.sqrt(self.n_units * self.p_con))
# 			w_out_init = np.random.normal(size=(self.n_outputs, self.n_units)) / 100
# 			taus_gaus_init = np.random.normal(size=(self.n_units, 1))
# 			bias_init = np.zeros(shape=(self.n_outputs, 1))
# 		np.random.seed(None)
# 		return w_init.astype(np.float32), w_out_init.astype(np.float32), taus_gaus_init.astype(np.float32), bias_init.astype(np.float32)
		
# 	'''
# 	Loss function
# 	'''
# 	@tf.function
# 	def loss_fcn(self, output, target):
# 		loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, target))))
# 		return loss
	
# 	'''
# 	Get trainable vars
# 	'''
# 	def get_trainable(self):
# 		return self.trainable_vars

# ########
# # MAIN #
# ########
# '''
# Settings
# 1. Number of units (n_units)
# 2. Probability of connection (p_con)
# 3. Probability of unit being inhibitory (p_inh)
# 4. Train mode setting (dotrain)
# 5. Test mode setting (dotest)
# 6. Restore mode setting (dorestore)
# 7. Model's random seed / run number (seed)
# 8. Which GPU to use or None for CPU (gpu)
# Example command:
# python Rnn.py 200 0.8 0.2 True False False 0 None
# '''
# # Model settings
# n_units = int(sys.argv[1])
# n_inputs = 1
# n_outputs = 1
# p_con = float(sys.argv[2])
# p_inh = float(sys.argv[3])
# g = 1.5
# taus = [4, 20] # Timesteps. 1 timestep = 5 ms
# seed = int(sys.argv[7])
# trial_len = 200 # Timesteps
# activation_fcn = tf.sigmoid

# # Training settings
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, name="adam")
# max_ep = 10000
# n_losses = 50
# threshold = 1

# # Test settings
# test_eps = 2

# # Gpu settings
# frac = 0.2
# gpu = str(sys.argv[8])

# # Gather hyperparameters of interest. INPUT LAST LINE MANUALLY!
# hyperparams = {"n_units":n_units, "n_inputs":n_inputs, "n_outputs":n_outputs,\
# 	"p_con":p_con, "p_inh":p_inh, "g":g, "taus":taus, "trial_len":trial_len,\
# 	"threshold":threshold, "n_losses":n_losses,\
# 	"activation_fcn":"sigmoid", "optimizer":"adam", "alpha":1e-3}

# # Mode
# dotrain, dotest, dorestore = [sys.argv[i].lower() == "true" for i in range(4,7)]

# # File structure, file names
# # root = "C:/Users/stefa/OneDrive/Rnn/ModelData"
# root = "/cnl/data/spate/Rnn/ModelData"
# cond = "Gonogo"
# run = f"Run_{seed:02}"
# cond_dir = root + "/" + cond
# run_dir = cond_dir + "/" + run
# h.make_dirs(cond_dir, run_dir) # Make sure there are directories before saving data

# '''
# Action
# '''
# # Configure gpu
# if gpu != None:
# 	config = h.set_gpu(gpu, frac)
# 	sess = InteractiveSession(config=config)

# # Create rnn object. Restore variables if restoring
# if dorestore:
# 	trainable_vars = h.load_trainable(run_dir)
# 	network = rnn(n_units, n_inputs, n_outputs, p_con, p_inh, g, taus, activation_fcn, seed=seed, restored_vars=trainable_vars)
# else:
# 	network = rnn(n_units, n_inputs, n_outputs, p_con, p_inh, g, taus, activation_fcn, seed=seed)

# # Train
# if dotrain:
# 	ep = 0
# 	met_threshold = False
# 	last_n_losses = [] # Keep last n losses in here
# 	saved_losses = [] # Save losses here
# 	while (ep < max_ep) & (not met_threshold): # Go till performance threshold met
# 		u, z = tasks.gonogo(trial_len, n_inputs, n_outputs) # Get stim, target
# 		loss = network.train(u, z, trial_len, optimizer) # Call training step
# 		last_n_losses.append(loss)
		
# 		# Wait at least 50 eps to start checking performance against threshold
# 		if len(last_n_losses) >= n_losses:
# 			if (sum(last_n_losses) / n_losses) < threshold: # Check if met threshold
# 				met_threshold = True
# 				print(f"Performance threshold met in {ep+1} episodes.")
# 				hyperparams['eps_to_threshold'] = ep + 1
				
# 				with open(run_dir + "/" + f"Met threshold in {ep+1} episodes.txt", "w") as f:
# 					f.write(f"{ep+1}")
				
# 				for var in network.get_trainable():
# 					np.savetxt(run_dir + "/" + f"{var.name[:-2]}.csv", var.numpy(), delimiter=",") # Must cut off ":#" from end of name
# 			else:
# 				last_n_losses = last_n_losses[1:]

# 		if ep % 20 == 0:
# 			saved_losses.append(loss)

# 		if ep % 100 == 0:
# 			print(f"Episode {ep+1}. Loss = {float(loss):.2f}")

# 		ep += 1

# 	# Save stuff
# 	with open(run_dir + "/" + f"hyperparams.csv", "w") as f:
# 		writer = csv.writer(f)
# 		for k,v in hyperparams.items():
# 			writer.writerow([k, v])
	
# 	np.savetxt(run_dir + "/" + "loss.csv", saved_losses, delimiter=",")

# # Test
# if dotest:
# 	u_list = []
# 	z_list = []
# 	output_list = []
# 	syncur_list = []

# 	for i in range(test_eps):
# 		u, z = tasks.gonogo(trial_len, n_inputs, n_outputs)
# 		output, syncur = network.simulate(u, trial_len)
# 		output, syncur = output.numpy(), syncur.numpy()
# 		output_list.append(output)
# 		syncur_list.append(syncur)
# 		u_list.append(u)
# 		z_list.append(z)

# 	# Save stuff
# 	np.savetxt(run_dir + "/" + "output.csv", np.vstack(output_list), delimiter=",")
# 	np.savetxt(run_dir + "/" + "syncur.csv", np.hstack(syncur_list), delimiter=",")
# 	np.savetxt(run_dir + "/" + "stim.csv", np.vstack(u_list), delimiter=",")
# 	np.savetxt(run_dir + "/" + "target.csv", np.vstack(z_list), delimiter=",")

# # End InteractiveSession if using gpu
# if gpu != None:
# 	sess.close()