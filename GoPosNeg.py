import csv
import sys
path = '/cnl/data/spate/Rnn'
if path not in sys.path:
	sys.path.append(path)
import numpy as np
import os
import tensorflow as tf
import Tasks as tasks
import Helpers as h
import Model
import ModelT
from tensorflow.compat.v1 import InteractiveSession

'''
Settings
1. Number of units (n_units)
2. Probability of connection (p_con)
3. Probability of unit being inhibitory (p_inh)
4. Number of inputs / outputs (n_inputs & n_outputs)
5. Train mode setting (dotrain)
6. Test mode setting (dotest)
7. Restore mode setting (dorestore)
8. Model's random seed / run number (seed)
9. Which GPU to use or None for CPU (gpu)
Example command:
python GoPosNeg.py 200 0.8 0.2 1 True True False 0 4
'''
# Model settings
taus_trainable = True # Allow taus to be trainable

if taus_trainable:
	rnn = ModelT.rnn # Which model; w/ trainable taus is ModelT.rnn; w/ fixed taus is Model.rnn
else:
	rnn - Model.rnn

n_units = int(sys.argv[1])
p_con = float(sys.argv[2])
dale = True # Impose Dale's law

if dale:
	p_inh = float(sys.argv[3])
else:
	p_inh = 0.0

g = 1.5
taus = [4, 20] # Taus range in timesteps. 1 timestep = 5 ms
seed = int(sys.argv[8])
activation_fcn = "sigmoid" # "tanh" / "sigmoid"

# Task settings
n_inputs = int(sys.argv[4])
n_outputs = int(sys.argv[4])
trial_len = 200 # Timesteps
stim_on = 50
stim_off = 75
prob_pulse = 0.5

# Training settings
alpha = 1e-3
batch_size = 64
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, name="adam") # INPUT MANUALLY TO hyperparams
max_ep = 20000
n_losses = 50 # Number episode losses to average over in evaluating performance criteria
threshold = 0.005 # Mean squared error

# Test settings
test_eps = 10

# Gpu settings
frac = 0.2
gpu = str(sys.argv[9])

# Gather hyperparameters of interest. INPUT LAST LINE MANUALLY!
hyperparams = {"n_units":n_units, "n_inputs":n_inputs, "n_outputs":n_outputs, "p_con":p_con,\
	"dale":dale, "p_inh":p_inh, "g":g, "taus_trainable":taus_trainable, "taus":taus, "trial_len":trial_len,\
	"stim_on":stim_on, "stim_off":stim_off, "threshold":threshold, "n_losses":n_losses, "alpha":alpha,\
	"batch_size":batch_size, "activation_fcn":activation_fcn, "optimizer":"adam"}

# Mode
dotrain, dotest, dorestore = [sys.argv[i].lower() == "true" for i in range(5,8)]

# File structure, file names
# root = "C:/Users/stefa/OneDrive/Rnn/ModelData"
root = "/cnl/data/spate/Rnn/ModelData"
cond = "GoPosNeg"
run = f"Run_{seed:02}"
cond_dir = root + "/" + cond
run_dir = cond_dir + "/" + run
h.make_dirs(cond_dir, run_dir) # Make sure there are directories before saving data

'''
Action
'''
# Configure gpu
if gpu != None:
	config = h.set_gpu(gpu, frac)
	sess = InteractiveSession(config=config)

# Create rnn object. Restore variables if restoring
if dorestore:
	trainable_vars = h.load_trainable(run_dir)
	network = rnn(n_units, n_inputs, n_outputs, p_con, dale, p_inh, g, taus, activation_fcn, seed=seed, restored_vars=trainable_vars)
else:
	network = rnn(n_units, n_inputs, n_outputs, p_con, dale, p_inh, g, taus, activation_fcn, seed=seed)

# Train
if dotrain:
	ep = 0
	met_threshold = False
	last_n_losses = [] # Keep last n losses in here
	saved_losses = [] # Save losses here
	while (ep < max_ep) & (not met_threshold): # Go till performance threshold met
		u, z = tasks.go_pos_neg(trial_len, stim_on, stim_off, n_inputs, n_outputs, prob_pulse, batch_size=batch_size) # Get stim, target
		loss = network.train(u, z, trial_len, optimizer, batch_size) # Call training step
		last_n_losses.append(loss)
		
		# Wait at least 50 eps to start checking performance against threshold
		if len(last_n_losses) >= n_losses:
			if (sum(last_n_losses) / n_losses) < threshold: # Check if met threshold
				met_threshold = True
				print(f"Performance threshold met in {ep+1} episodes.")
				hyperparams['eps_to_threshold'] = ep + 1
				
				with open(run_dir + "/" + f"Met threshold in {ep+1} episodes.txt", "w") as f:
					f.write(f"{ep+1}")
			else:
				last_n_losses = last_n_losses[1:]

		if ep % 20 == 0:
			saved_losses.append(loss)

		if ep % 100 == 0:
			print(f"Episode {ep+1}. Loss = {float(loss):.4f}")

		ep += 1
	
	# Save stuff
	for var in network.get_trainable():
		np.savetxt(run_dir + "/" + f"{var.name[:-2]}.csv", var.numpy(), delimiter=",") # Must cut off ":#" from end of name
	
	with open(run_dir + "/" + "hyperparams.csv", "w") as f:
		writer = csv.writer(f)
		for k,v in hyperparams.items():
			writer.writerow([k, v])
	
	np.savetxt(run_dir + "/" + "loss.csv", saved_losses, delimiter=",")

# Test
if dotest:
	u_list = []
	z_list = []
	output_list = []
	syncur_list = []

	for i in range(test_eps):
		u, z = tasks.go_pos_neg(trial_len, stim_on, stim_off, n_inputs, n_outputs, prob_pulse, batch_size=1) # Get stim, target
		output, syncur = network.simulate(u, trial_len)
		output, syncur = output.numpy(), syncur.numpy()
		output_list.append(output)
		syncur_list.append(syncur)
		u_list.append(u.reshape(n_inputs, trial_len))
		z_list.append(z.reshape(n_outputs, trial_len))

	# Save stuff
	np.savetxt(run_dir + "/" + "output.csv", np.vstack(output_list), delimiter=",")
	np.savetxt(run_dir + "/" + "syncur.csv", np.hstack(syncur_list), delimiter=",")
	np.savetxt(run_dir + "/" + "stim.csv", np.vstack(u_list), delimiter=",")
	np.savetxt(run_dir + "/" + "target.csv", np.vstack(z_list), delimiter=",")

# End InteractiveSession if using gpu
if gpu != None:
	sess.close()