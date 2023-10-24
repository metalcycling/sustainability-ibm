"""
Convergence comparison

%load_ext autoreload
%autoreload 2
"""

# %% Modules

import json
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

def parse_log(filename):
    """
    Parse training report from logs
    """
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

def parse_json(filename):
    """
    Parse training report in JSON format
    """
    fileptr = open(filename, "r")
    lines = json.load(fileptr)
    fileptr.close()

    data = {}
    data["training"] = { "loss": [], "step": [], "speed": [] }
    data["validation"] = { "loss": [], "step": [], "speed": [] }

    for line in lines:
        if "trainloss" in line:
            data["training"]["step"].append(line["step"])
            data["training"]["loss"].append(line["trainloss"])
            data["training"]["speed"].append(line["speed"])

        if "valloss" in line:
            data["validation"]["step"].append(line["step"])
            data["validation"]["loss"].append(line["valloss"])

    data["training"]["step"] = np.array(data["training"]["step"])
    data["training"]["loss"] = np.array(data["training"]["loss"])
    data["training"]["speed"] = np.array(data["training"]["speed"])

    data["validation"]["step"] = np.array(data["validation"]["step"])
    data["validation"]["loss"] = np.array(data["validation"]["loss"])

    num_steps = data["validation"]["step"][0] // data["training"]["step"][0]
    interval_training = data["training"]["step"][1:] - data["training"]["step"][:-1]
    interval_training = np.hstack((interval_training[0], interval_training))
    interval_validation = data["validation"]["step"][1:] - data["validation"]["step"][:-1]
    interval_validation = np.hstack((interval_validation[0], interval_validation))

    for idx in range(len(data["validation"]["step"])):
        jdx_start = (idx + 0) * num_steps
        jdx_stop = (idx + 1) * num_steps
        data["validation"]["speed"].append(np.sum(data["training"]["speed"][jdx_start:jdx_stop] * interval_training[jdx_start:jdx_stop]) / interval_validation[idx])

    data["validation"]["speed"] = np.array(data["validation"]["speed"])

    return data

# %% Main program

if __name__ == "__main__":

    # %% Read wiki training

    filename = "../../March_06_2023/Data/batch_size_8/log_main.json"
    wiki = parse_json(filename)

    # %% Read cbt training

    filename = "../Data/roberta-8.dat"
    cbt = parse_log(filename)

    # %% Visualization

    num_lines = 40000

    plt.figure(figsize = (12, 8))
    plt.title("Comparison of RoBERTa convergence with 2 datasets", fontsize = title_size)
    plt.xlabel("Number of steps", fontsize = label_size)
    plt.ylabel("Training loss", fontsize = label_size)

    losses = wiki["training"]["loss"][:num_lines]
    steps = wiki["training"]["step"][:num_lines]
    plt.plot(steps, losses, linewidth = line_width, label = "Wikipedia")

    losses = cbt["trainlosses"][:num_lines]
    steps = cbt["steps"][:num_lines]
    plt.plot(steps, losses, linewidth = line_width, label = "Children's Book Test")

    plt.xlim(steps[0], steps[-1])
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)

    plt.grid()
    plt.legend(fontsize = legend_size)

    plt.savefig("../Figures/comparison.pdf", bbox_inches = "tight")

    plt.show()

# %% End of program
