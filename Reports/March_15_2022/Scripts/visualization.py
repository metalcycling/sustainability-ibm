# Modules
import os
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Text parameters
title_size = 19
label_size = 18
ticks_size = 16
legend_size = 17
text_size = 15
line_width = 2

# Functions
def time_breakdown(filename):
    proc = subprocess.Popen(["grep", "s ( ", filename], stdout = subprocess.PIPE)
    breakdown = []

    for idx, line in enumerate(proc.stdout.readlines()):
        if idx > 0:
            breakdown.append(float(line.decode("utf-8").split(":")[1].split("s")[0]))

    return breakdown

# Parsing
directory = "../Data"

bar_width = 0.15
bar_offset = 0.5
x_ticks = ["Reading", "Loading", "Prediction", "Backpropagation"]

fig, ax = plt.subplots(1, figsize = (12, 8))
#plt.title("Poisson solver time for '%s'" % (mesh_data[mesh_name]["label"]), fontsize = title_size)
#plt.xlabel("Number of processors", fontsize = label_size)
plt.ylabel("Time (seconds)", fontsize = label_size)

for idx, device in enumerate(["cpu", "gpu"]):
    filename = "%s/%s.dat" % (directory, device)
    breakdown = time_breakdown(filename)

    plt.bar(np.arange(len(breakdown)) + idx * bar_width, breakdown, width = bar_width, label = device, zorder = 3)

plt.grid(zorder = 0)
plt.legend(fontsize = legend_size)
plt.xticks(np.arange(len(x_ticks)) + (idx + 0.5) * bar_width, x_ticks, fontsize = ticks_size)
plt.yticks(fontsize = ticks_size)
plt.show()
