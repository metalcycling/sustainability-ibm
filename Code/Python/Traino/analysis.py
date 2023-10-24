import numpy as np
import matplotlib.pyplot as plt

# Formatting
title_size = 17
label_size = 16
ticks_size = 15
legend_size = 15

# Parsing
data = np.loadtxt("results.dat")
#data = np.loadtxt("results_v1.dat")
num_features_list = np.unique(data[:, 0]).astype(int)
batch_size_list = np.unique(data[:, 1]).astype(int)

# Plotting
bar_width = 0.10
bar_offset = 0.5
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

#"""
fig, ax = plt.subplots(figsize = (24, 8))
plt.title("Reading time of CFSHM vs Numpy")
plt.xlabel("Batch size", fontsize = label_size)
plt.ylabel("Time (seconds)", fontsize = label_size)

for i, num_features in enumerate(num_features_list):
    sample = data[(i + 0) * len(batch_size_list) : (i + 1) * len(batch_size_list)]

    plt.bar(np.arange(len(batch_size_list)) * (1.0 + bar_offset) + i * bar_width, sample[:, 3], color = colors[i % len(colors)], hatch = "x" * 8, width = bar_width, zorder = 3)
    plt.bar(np.arange(len(batch_size_list)) * (1.0 + bar_offset) + i * bar_width, sample[:, 2], color = colors[i % len(colors)], width = bar_width, zorder = 3, label = "$n = %d$" % (num_features))


ax.set_yscale("log")

plt.grid(zorder = 0)
plt.legend(loc = 2, fontsize = legend_size)
plt.xticks(np.arange(len(batch_size_list)) * (1.0 + bar_offset) + np.round(0.5 * (len(num_features_list) - 1)) * bar_width, batch_size_list, fontsize = ticks_size)
plt.yticks(fontsize = ticks_size)
#"""

#"""
fig, ax = plt.subplots(figsize = (24, 8))
plt.title("Percentage overhead of CFSHM over Numpy")
plt.xlabel("Batch size", fontsize = label_size)
plt.ylabel("Percentage", fontsize = label_size)

for i, num_features in enumerate(num_features_list):
    sample = data[(i + 0) * len(batch_size_list) : (i + 1) * len(batch_size_list)]

    plt.bar(np.arange(len(batch_size_list)) * (1.0 + bar_offset) + i * bar_width, 100.0 * (sample[:, 3] - sample[:, 2]) / sample[:, 2], color = colors[i % len(colors)], width = bar_width, zorder = 3, label = "$n = %d$" % (num_features))

plt.grid(zorder = 0)
plt.legend(loc = 2, fontsize = legend_size)
plt.xticks(np.arange(len(batch_size_list)) * (1.0 + bar_offset) + np.round(0.5 * (len(num_features_list) - 1)) * bar_width, batch_size_list, fontsize = ticks_size)
plt.yticks(fontsize = ticks_size)
#"""

plt.show()
