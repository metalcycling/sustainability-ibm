"""
Dynamic batch

%load_ext autoreload
%autoreload 2
"""

# %% Modules

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# %% Formatting

title_size = 20
label_size = 17
legend_size = 16
line_width = 2
ticks_size = 15

# %% Functions

def parse_data(filename):
    fileptr = open(filename, "r")
    steps = []
    trainlosses = []
    speeds = []

    for idx, line in enumerate(fileptr):
        words = line.split()

        if idx % 4 == 0:
            steps.append(int(words[-1]))

        elif idx % 4 == 1:
            trainlosses.append(float(words[-1]))

        elif idx % 4 == 2:
            speeds.append(float(words[-1]))

    fileptr.close()

    return { "steps": steps, "trainlosses": trainlosses, "speeds": speeds }

# %% Main program

if __name__ == "__main__":
    directory = "../Data"
    batch_sizes = ["8", "16", "32", "64", "dynamic"]
    data = { batch_size: parse_data("%s/batch_size_%s.dat" % (directory, batch_size)) for batch_size in batch_sizes }

    plt.figure(figsize = (12, 8))
    plt.title("Convergence rates", fontsize = title_size)
    plt.xlabel("Number of steps", fontsize = label_size)
    plt.ylabel("Training Loss", fontsize = label_size)

    for batch_size in batch_sizes:
        plt.plot(data[batch_size]["steps"], data[batch_size]["trainlosses"], linewidth = line_width, label = "B = %s" % (batch_size))

    plt.grid()
    plt.legend(fontsize = legend_size)
    plt.xlim(100.0, 4000.0)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)

    plt.savefig("../Figures/dynamic.pdf", bbox_inches = "tight")

    """
    plt.figure(figsize = (12, 8))
    plt.title("Convergence rates", fontsize = title_size)
    plt.xlabel("Number of steps", fontsize = label_size)
    plt.ylabel("Training Loss", fontsize = label_size)

    for batch_size in batch_sizes:
        plt.plot(data[batch_size]["steps"], data[batch_size]["trainlosses"], linewidth = line_width, label = "B = %s" % (batch_size))

    plt.grid()
    plt.legend(fontsize = legend_size)
    plt.xlim(100.0, 4000.0)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    """

    plt.show()

# %% End of program
