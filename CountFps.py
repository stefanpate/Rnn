import os
import numpy as np

root = "C:/Users/stefa/OneDrive/Rnn/ModelData/"
conds = ["Gonogo", "FlipFlop", "GoPosNeg"]
n_units = 200

os.chdir(root)
for cond in conds:
    cond_dir = root + cond
    runs = os.listdir(cond_dir)
    for run in runs:
        run_dir = cond_dir + "/" + run
        if "fps.csv" in os.listdir(run_dir):
            fps = np.loadtxt(run_dir + "/" + "fps.csv", delimiter=",").reshape(-1,n_units)
            n_fps = fps.shape[0]
            with open(run_dir + "/" + f"{n_fps}_fps.txt", "w") as f:
                f.write(f"{n_fps} fixed points.")
        if "sub_fps.csv" in os.listdir(run_dir):
            sub_fps = np.loadtxt(run_dir + "/" + "sub_fps.csv", delimiter=",").reshape(-1,n_units)
            n_sub_fps = sub_fps.shape[0]
            with open(run_dir + "/" + f"{n_sub_fps}_sub_fps.txt", "w") as f:
                f.write(f"{n_fps} sub fixed points.")