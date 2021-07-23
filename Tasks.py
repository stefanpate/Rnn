import numpy as np
'''
Go / No-go task
Parameters:
	- trial_len: Length of trial (int)
	- stim_on: Timestep to start the stim (int)
	- stim_off: Timestep to stop the stim (int)
	- n_inputs: Number of inputs / stimuli given simultaneously to the network (int)
	- n_outputs: Number of outputs required of the network (int)
	- prob_pulse: Probability that a given trial is a "go-trial" (float [0,1])
	- batch_size: Number of trials in a batch (int)
Returns:
	- u: Stimulus (3D numpy array)
	- z: Target (3D numpy array)
'''
def gonogo(trial_len, stim_on, stim_off, n_inputs, n_outputs, prob_pulse, batch_size=1):
	u = np.zeros((n_inputs, trial_len, batch_size))
	z = np.zeros((n_outputs, trial_len, batch_size))
	pulse_idxs = np.where(np.random.rand(batch_size) < prob_pulse)[0] # Boolean array where to put the pulses
	u[:, stim_on:stim_off, pulse_idxs] += 1
	z[:, stim_off:, pulse_idxs] += 1
	return u, z

'''
Like go / no-go except you go in the direction of the stimulus (either +1 or -1)
Parameters:
	- trial_len: Length of trial (int)
	- stim_on: Timestep to start the stim (int)
	- stim_off: Timestep to stop the stim (int)
	- n_inputs: Number of inputs / stimuli given simultaneously to the network (int)
	- n_outputs: Number of outputs required of the network (int)
	- prob_pulse: Probability that a given trial is a "go-trial" (float [0,1])
	- batch_size: Number of trials in a batch (int)
Returns:
	- u: Stimulus (3D numpy array)
	- z: Target (3D numpy array)
'''
def go_pos_neg(trial_len, stim_on, stim_off, n_inputs, n_outputs, prob_pulse, batch_size=1):
	u = np.zeros((n_inputs, trial_len, batch_size))
	z = np.zeros((n_outputs, trial_len, batch_size))
	pos_pulse_mask = np.random.rand(batch_size) < prob_pulse # Boolean array where to put the postivie pulses
	neg_pulse_mask = ~pos_pulse_mask
	u[:, stim_on:stim_off, pos_pulse_mask] += 1
	u[:, stim_on:stim_off, neg_pulse_mask] -= 1
	z[:, stim_off:, pos_pulse_mask] += 1
	z[:, stim_off:, neg_pulse_mask] -= 1
	return u, z

'''
N bit flip flop task
Parameters:
	- trial_len: Length of trial (int)
	- n_inputs: Number of inputs / stimuli given simultaneously to the network (int)
	- n_outputs: Number of outputs required of the network (int)
	- batch_size: Number of trials in a batch (int)
Returns:
	- u: Stimulus (3D numpy array)
	- z: Target (3D numpy array)
'''
def n_bit_flip_flop(trial_len, n_inputs, n_outputs, batch_size=1):
	if n_inputs != n_outputs:
		raise ValueError("Number of inputs and outputs must match.")
	
	np.random.seed(None)
	prob_pulse = 0.05
	u = np.zeros((n_inputs, trial_len, batch_size))
	z = np.zeros((n_outputs, trial_len, batch_size))
	last_pulse = np.zeros((n_inputs,batch_size))

	for i in range(trial_len):
		do_pulse = np.random.rand(n_inputs, batch_size) < prob_pulse # Chosen probability of pulsing
		pos_given_pulse = np.random.rand(n_inputs, batch_size) < 0.5 # 1/2 probability +1, given a pulse
		pulse_i = np.where(do_pulse & pos_given_pulse, 1, 0)
		pulse_i = np.where(do_pulse & ~pos_given_pulse, -1, pulse_i)
		u[:,i,:] = pulse_i
		z[:,i,:] = last_pulse
		last_pulse = np.where(pulse_i != 0, pulse_i, last_pulse)
	
	return u, z

'''
Off-shoot of n bit flip flop where two specified pulses are given. Example usage
is to put the network through all possible state transitions (e.g., [1,1,1] -> [1,1,-1])
Parameters:
	- input_one: First input to network (1D array [n_inputs,])
	- input_two: Second input to network (1D array [n_inputs,])
	- trial_len: Length of trial (int)
	- null_time: Number of timesteps between start of the trial and the first input (int)
	- n_inputs: Number of inputs / stimuli given simultaneously to the network. Assumes
				number of inputs and outputs matches (int)
	- batch_size: Number of trials in a batch (int)
Returns:
	- u: Stimulus (3D numpy array)
	- z: Target (3D numpy array)
'''
def two_input_flip_flop(input_one, input_two, trial_len, null_time, n_inputs, batch_size=1):
	input_one = input_one.reshape(-1,1)
	input_two = input_two.reshape(-1,1)
	u = np.zeros((n_inputs, trial_len, batch_size))
	z = np.zeros((n_inputs, trial_len, batch_size))

	if (trial_len - null_time) % 2 != 0: # Ensure the remaining time after null time is even
		null_time -= 1

	half_remaining_time = (trial_len - null_time) // 2
	t_1 = null_time
	t_2 = t_1 + half_remaining_time
	target_one = np.repeat(input_one, half_remaining_time, axis=1).reshape(-1, half_remaining_time, 1)
	target_two = np.where(input_two != 0, input_two, input_one)
	target_two = np.repeat(target_two, half_remaining_time, axis=1).reshape(-1, half_remaining_time, 1)
	u[:,t_1,:] = input_one
	u[:,t_2,:] = input_two
	z[:,t_1:t_2,:] = target_one
	z[:,t_2:,:] = target_two
	return u, z