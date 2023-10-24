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
x_ticks = ["File reading", "GPU Loading", "Inference", "Backpropagation"]

"""
for width in [8, 16]:
    for batch_size in [8, 64, 512]:
        fig, ax = plt.subplots(1, figsize = (12, 8))
        #plt.title("Poisson solver time for '%s'" % (mesh_data[mesh_name]["label"]), fontsize = title_size)
        #plt.xlabel("Number of processors", fontsize = label_size)
        plt.ylabel("Time (seconds)", fontsize = label_size)

        for idx, device in enumerate(["cpu", "gpu"]):
            filename = "%s/%s_%dx%d_B%d.dat" % (directory, device, width, width, batch_size)
            breakdown = time_breakdown(filename)
            #print(breakdown)
            #quit()

            plt.bar(np.arange(len(breakdown)) + idx * bar_width, breakdown, width = bar_width, label = device, zorder = 3)

        plt.grid(zorder = 0)
        plt.legend(fontsize = legend_size)
        plt.xticks(np.arange(len(x_ticks)) + (idx + 0.5) * bar_width, x_ticks, fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)

"""

#"""
for device in ["cpu", "gpu"]:
    for width in [8, 16]:
        fig, ax = plt.subplots(1, figsize = (12, 8))
        plt.title("Run with '%s' and 'w = %d'" % (device.upper(), width), fontsize = title_size)
        plt.ylabel("Time (seconds)", fontsize = label_size)

        for idx, batch_size in enumerate([8, 64, 512, 1024, 2048]):
            filename = "%s/%s_%dx%d_B%d.dat" % (directory, device, width, width, batch_size)
            breakdown = time_breakdown(filename)

            plt.bar(np.arange(len(breakdown)) + idx * bar_width, breakdown, width = bar_width, label = "$B = %d$" % (batch_size), zorder = 3)

        plt.grid(zorder = 0)
        plt.legend(loc = 2, fontsize = legend_size)
        plt.xticks(np.arange(len(x_ticks)) + (2.0 - 0.0) * bar_width, x_ticks, fontsize = ticks_size)
        plt.ylim(0.0, 320.0)
        plt.yticks(fontsize = ticks_size)
        plt.savefig("../Figures/%s_%dx%d_full.pdf" % (device, width, width), bbox_inches = "tight")
#"""

"""
for device in ["cpu", "gpu"]:
    for width in [8, 16]:
        fig, ax = plt.subplots(1, figsize = (12, 8))
        plt.title("Run with '%s' and 'w = %d'" % (device.upper(), width), fontsize = title_size)
        plt.ylabel("Time (seconds)", fontsize = label_size)

        for idx, batch_size in enumerate([64, 512, 1024, 2048]):
            filename = "%s/%s_%dx%d_B%d.dat" % (directory, device, width, width, batch_size)
            breakdown = time_breakdown(filename)

            plt.bar(np.arange(len(breakdown)) + idx * bar_width, breakdown, width = bar_width, label = "$B = %d$" % (batch_size), zorder = 3)

        plt.grid(zorder = 0)
        plt.legend(fontsize = legend_size)
        plt.xticks(np.arange(len(x_ticks)) + (2.0 - 0.5) * bar_width, x_ticks, fontsize = ticks_size)
        plt.ylim(0.0, 45.0)
        plt.yticks(fontsize = ticks_size)
        plt.savefig("../Figures/%s_%dx%d_reduced.pdf" % (device, width, width), bbox_inches = "tight")
#"""

plt.show()
