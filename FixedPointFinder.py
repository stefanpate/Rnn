import os
import sys
import numpy as np
import tensorflow as tf
import Helpers as h
from Model import rnn
from tensorflow.compat.v1 import InteractiveSession

# @tf.function
def find_fps(network, x, u, max_iters, tol_q, tol_dq, optimizer, max_norm):
	tol_q, tol_dq = tf.constant(tol_q), tf.constant(tol_dq)
	for i in range(max_iters):
		with tf.GradientTape() as tape:
			tape.watch([x])
			x_dot = network.f_x(x, u)
			q = tf.multiply(0.5, tf.reduce_sum(tf.square(x_dot)))

			if i > 0:
				dq = tf.abs(q - q_prev)
				if (q <= tol_q) & (dq <= tol_dq):
					print("Found fixed point")
					x = tf.reshape(x, shape=(-1,1))
					return x

			if i % 250 == 0:
				print(f"Iteration: {i}, q: {q}", end="\r")
			
			gradients = tape.gradient(q, [x])
			gradients = [tf.clip_by_norm(g, max_norm) for g in gradients]
			optimizer.apply_gradients(zip(gradients, [x]))
			q_prev = q

'''
Settings
1. Number of units (n_units)
2. Probability of connection (p_con)
3. Probability of unit being inhibitory (p_inh)
4. Number of bits to be flip flopped (n_inputs & n_outputs)
5. Model's random seed / run number (seed)
6. Which GPU to use or None for CPU (gpu)
Example command:
python FixedPointFinder.py 200 0.8 0.0 3 6 1
'''
# Model settings
n_units = int(sys.argv[1])
n_inputs = int(sys.argv[4])
n_outputs = int(sys.argv[4])
p_con = float(sys.argv[2])
dale = False
if dale:
	p_inh = float(sys.argv[3])
else:
	p_inh = 0.0
g = 1.5
taus = [1, 1] # Timesteps. 1 timestep = 5 ms
seed = int(sys.argv[5])
trial_len = 100 # Timesteps
activation_fcn = tf.tanh

# Optimization settings
tol_q = 1e-12 # q must be at least this small to stop optimization AND
tol_dq = 1e-20 # AND change in q must be this small
tol_unique = 1e-3 # Unique fixed point must be this different (2 norm) from others found
alpha = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, name="adam")
max_norm  = 40.
max_iters = 5000
n_init_states = 600
batch_size = 1

# Gpu settings
frac = 0.2
gpu = str(sys.argv[6])

# File structure, file names
# root = "C:/Users/stefa/OneDrive/Rnn/ModelData"
root = "/cnl/data/spate/Rnn/ModelData"
cond = "FlipFlop"
run = f"Run_{seed:02}"
cond_dir = root + "/" + cond
run_dir = cond_dir + "/" + run

'''
Action
'''
# Configure gpu
if gpu != None:
	config = h.set_gpu(gpu, frac)
	sess = InteractiveSession(config=config)

trainable_vars = h.load_trainable(run_dir)
network = rnn(n_units, n_inputs, n_outputs, p_con, dale, p_inh, g, taus, activation_fcn, seed=seed, restored_vars=trainable_vars)

# Run optimization
init_state = np.random.rand(n_units, batch_size)
u = np.zeros((n_inputs, batch_size))
fps = []
os.chdir(run_dir)
syncur = np.loadtxt("syncur.csv", delimiter=",")
syncur_idxs = np.random.choice(syncur.shape[1], n_init_states)
for i,idx in enumerate(syncur_idxs):
	init_state = syncur[:,idx].reshape(n_units, batch_size)
	x = tf.Variable(init_state, name="x", shape=(n_units, batch_size))
	print(f"\nInit state: {i+1}")		
	fp = find_fps(network, x, u, max_iters, tol_q, tol_dq, optimizer, max_norm)
	if fp != None:
		fps.append(fp.numpy())
if len(fps) > 0:
	fps = np.hstack(fps)
	np.savetxt(run_dir + "/" + "fps.csv", fps, delimiter=",")

# End InteractiveSession if using gpu
if gpu != None:
	sess.close()