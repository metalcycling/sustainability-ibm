"""
Visualization script for RoBERTa results

%load_ext autoreload
%autoreload 2
"""

# %% Modules

import json
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime

# %% Formatting

title_size = 26
label_size = 24
ticks_size = 24
legend_size = 21
line_width = 3
marker_size = 12

# %% Functions

def get_power(filename):
    """
    Parse power data
    """
    fileptr = open(filename, "r")
    reference_time = None
    time = 0.0
    power_data = []

    for line in fileptr:
        sample = line.split()

        if reference_time == None:
            reference_time = datetime.strptime(sample[0], "%Y-%m-%d-%H:%M:%S.%f-%Z")
            power_data.append([0.0] + list(map(float, sample[1:])))
        else:
            lapsed_time = datetime.strptime(sample[0], "%Y-%m-%d-%H:%M:%S.%f-%Z") - reference_time
            power_data.append([lapsed_time.total_seconds()] + list(map(float, sample[1:])))

    fileptr.close()

    return np.array(power_data)

def get_training(filename):
    """
    Parse training data
    """
    fileptr = open(filename, "r")
    data = json.load(fileptr)
    fileptr.close()

    step = []
    train_loss = []
    speed = []

    for idx in range(len(data) - 1, - 1, - 1):
        if "valloss" in data[idx] or "total_hours" in data[idx] or "msg" in data[idx]:
            continue
        elif "step" not in data[idx]:
            break

        step.append(data[idx]["step"])
        train_loss.append(data[idx]["trainloss"])
        speed.append(data[idx]["speed"])

    return { "step": np.array(step[::-1]), "trainloss": np.array(train_loss[::-1]), "speed": np.array(speed[::-1]) }

# %% Main program

if __name__ == "__main__":
    directory = "Data"
    batch_sizes = [8, 16, 32, 64]
    powers = { batch_size: get_power("%s/batch_size_%d/power.dat" % (directory, batch_size)) for batch_size in batch_sizes }
    trainings = { batch_size: get_training("%s/batch_size_%d/log_main.json" % (directory, batch_size)) for batch_size in batch_sizes }
    time_in_hours = True

    # %%

    def time_vs_power():
        plt.figure(figsize = (26, 18))
        plt.title("Power data for the complete/unstopped training", fontsize = title_size)
        plt.xlabel("Lapsed hours" if time_in_hours else "Lapsed seconds", fontsize = label_size)
        plt.ylabel("Power (Watts)", fontsize = label_size)

        xmax = 0.0

        for batch_size in batch_sizes:
            power = powers[batch_size]

            if time_in_hours:
                plt.plot(power[:, 0] / 3600.0, power[:, 1:].sum(axis = 1), linewidth = line_width, label = "B = %d" % (batch_size))
            else:
                plt.plot(power[:, 0], power[:, 1:].sum(axis = 1), linewidth = line_width, label = "B = %d" % (batch_size))

            xmax = max(xmax, power[-1, 0] / 3600.0 if time_in_hours else power[-1, 0])

        plt.grid()
        plt.xlim(xmin = 0.0, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

    time_vs_power()
    plt.show()

    # %%

    def time_vs_energy():
        plt.figure(figsize = (26, 18))
        plt.title("Energy consumed for the complete/unstopped training", fontsize = title_size)
        plt.xlabel("Lapsed hours" if time_in_hours else "Lapsed seconds", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmax = 0.0

        for batch_size in batch_sizes:
            power = powers[batch_size]
            joules = sp.integrate.cumtrapz(power[:, 1:].sum(axis = 1), power[:, 0] / 3600.0, initial = 0.0)

            if time_in_hours:
                plt.plot(power[:, 0] / 3600.0, joules, linewidth = line_width, label = "B = %d" % (batch_size))
            else:
                plt.plot(power[:, 0], joules, linewidth = line_width, label = "B = %d" % (batch_size))

            xmax = max(xmax, power[-1, 0] / 3600.0 if time_in_hours else power[-1, 0])

        plt.grid()
        plt.xlim(xmin = 0.0, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

    time_vs_energy()
    plt.show()

    # %%

    def time_vs_energy_converged(threshold, idle_power):
        plt.figure(figsize = (26, 18))
        plt.title("Energy consumed to achieve a train loss of '%1.2f'" % (threshold), fontsize = title_size)
        plt.xlabel("Lapsed hours" if time_in_hours else "Lapsed seconds", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmax = 0.0

        for batch_size in batch_sizes:
            power = powers[batch_size]
            training = trainings[batch_size]
            timestamps = power[:, 0]
            wattage = power[:, 1:].sum(axis = 1)

            # Get completion time based on the threshold
            stop_idx = np.where(training["trainloss"] < threshold)[0][0]
            steps = training["step"][:stop_idx]
            speed = training["speed"][:stop_idx]
            interval = steps[1:] - steps[:-1]
            interval = np.hstack((interval[0], interval))
            total_time = np.sum(interval * speed)

            # Compute energy for the duration of the training until convergence
            start_idx = np.where(wattage >= idle_power)[0][0]
            start_time = timestamps[start_idx]
            stop_time = start_time + total_time
            stop_idx = np.where(timestamps >= stop_time)[0][0]

            joules = sp.integrate.cumtrapz(wattage[start_idx:stop_idx], (timestamps[start_idx:stop_idx] - start_time) / 3600.0, initial = 0.0)

            if time_in_hours:
                plt.plot((timestamps[start_idx:stop_idx] - start_time) / 3600.0, joules, linewidth = line_width, label = "B = %d" % (batch_size))
            else:
                plt.plot((timestamps[start_idx:stop_idx] - start_time), joules, linewidth = line_width, label = "B = %d" % (batch_size))

            xmax = max(xmax, stop_time / 3600.0 if time_in_hours else stop_time)

        plt.grid()
        plt.xlim(xmin = 0.0, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

    threshold = 2.2
    idle_power = 520.0
    time_vs_energy_converged(threshold, idle_power)
    plt.show()

    # %%

    def accuracy_vs_energy(idle_power):
        xmin = np.inf
        xmax = 0.0

        plt.figure(figsize = (26, 18))
        plt.title("Energy as a function of training accuracy", fontsize = title_size)
        plt.xlabel("Training loss", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        for batch_size in batch_sizes:
            power = powers[batch_size]
            training = trainings[batch_size]
            timestamps = power[:, 0]
            wattage = power[:, 1:].sum(axis = 1)

            loss = training["trainloss"].copy()
            loss.sort()
            thresholds = np.linspace(loss[-2], loss[1], 200)

            joules = []

            for threshold in thresholds:
                stop_idx = np.where(training["trainloss"] < threshold)[0][0]
                steps = training["step"][:stop_idx]
                speed = training["speed"][:stop_idx]
                interval = steps[1:] - steps[:-1]
                interval = np.hstack((interval[0], interval))
                total_time = np.sum(interval * speed)

                start_idx = np.where(wattage >= idle_power)[0][0]
                start_time = timestamps[start_idx]
                stop_time = start_time + total_time
                stop_idx = np.where(timestamps >= stop_time)[0][0]

                joules.append(sp.integrate.trapz(wattage[start_idx:stop_idx], (timestamps[start_idx:stop_idx] - start_time) / 3600.0))

            plt.semilogy(thresholds, joules, linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, thresholds[-1])
            xmax = max(xmax, thresholds[0])

        plt.grid()
        plt.xlim(xmin = xmax, xmax = xmin)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

    idle_power = 520.0
    accuracy_vs_energy(idle_power)
    plt.show()

    # %%

    def accuracy_vs_time():
        plt.figure(figsize = (26, 18))
        plt.title("Total execution time as a function of training accuracy", fontsize = title_size)
        plt.xlabel("Training loss", fontsize = label_size)
        plt.ylabel("Execution time (hours)", fontsize = label_size)

        xmin = np.inf
        xmax = 0.0

        for batch_size in batch_sizes:
            training = trainings[batch_size]
            interval = training["step"][1:] - training["step"][:-1]
            interval = np.hstack((interval[0], interval))
            speed = training["speed"]
            total_time = np.cumsum(interval * speed)

            plt.semilogy(training["trainloss"], total_time / 3600.0, linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, training["trainloss"].min())
            xmax = max(xmax, training["trainloss"].max())

        plt.grid()
        plt.xlim(xmin = xmax, xmax = xmin)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

    accuracy_vs_time()
    plt.show()

    # %%

    def convergence_history():
        plt.figure(figsize = (26, 18))
        plt.title("Convergece of the training as a function of the number of steps", fontsize = title_size)
        plt.xlabel("Step", fontsize = label_size)
        plt.ylabel("Training loss", fontsize = label_size)

        for batch_size in batch_sizes:
            training = trainings[batch_size]
            plt.plot(training["step"], training["trainloss"], linewidth = line_width, label = "B = %d" % (batch_size))

        plt.grid()
        plt.xlim(xmin = 10.0, xmax = max([trainings[batch_size]["step"][-1] for batch_size in batch_sizes]))
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

    convergence_history()
    plt.show()

    # %%

    def training_speed_history():
        plt.figure(figsize = (26, 18))
        plt.title("Training speed as a function of the number of steps", fontsize = title_size)
        plt.xlabel("Step", fontsize = label_size)
        plt.ylabel("Training speed", fontsize = label_size)

        for batch_size in batch_sizes:
            training = trainings[batch_size]
            plt.plot(training["step"], training["speed"], linewidth = line_width, label = "B = %d" % (batch_size))

        plt.grid()
        plt.xlim(xmin = 10.0, xmax = max([trainings[batch_size]["step"][-1] for batch_size in batch_sizes]))
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

    training_speed_history()
    plt.show()

# %% End of program
    
    idle_power = 520.0
    accuracy_vs_energy(idle_power)
    accuracy_vs_time()

    plt.show()

# %% End of program
