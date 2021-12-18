import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation

# Random FF net params
n_units = 25
n_inputs = 1
w_in = np.random.normal(size=(n_units,n_inputs))

# Get heatmap data
start = np.zeros((1, 20))
end = np.zeros((1, 30))
t = np.linspace(0, 2 * np.pi * 4, 50)
sine = np.sin(t).reshape(1, -1)
input = np.concatenate([start, sine, end], axis=1)

xs = np.matmul(w_in, input)
rs = np.tanh(xs)
data_for_plot = rs.reshape(5, 5, -1)

# Fcn to get input data
# def sine_input(tstart, tend, dt, n_pers)

fig, ax = plt.subplots(1, 2)


def init():
    sns.heatmap(np.zeros((5,5)), vmax=0.8, cbar=False, ax=ax[1])

def animate(i):
    data = data_for_plot[:,:,i]
    sns.heatmap(data, vmax=0.8, cbar=False, ax=ax[1])

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, repeat=False)
plt.show()

