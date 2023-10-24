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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# %% Formatting

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
presentation_mode = False
show_plots_individually = True

if presentation_mode:
    figsize = (26, 18)
    title_size = 26
    label_size = 24
    ticks_size = 24
    legend_size = 21
    line_width = 3
    marker_size = 44
    alpha = 0.3

else:
    figsize = (12, 8)
    title_size = 22
    label_size = 20
    ticks_size = 20
    legend_size = 17
    line_width = 3
    marker_size = 40
    alpha = 0.3

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

    return np.array(power_data, dtype = float)

def get_report(filename):
    """
    Parse training report
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

def augment_data(power, report, kernel_power= 1500.0):
    """
    Augment the data collected by the training script
    """

    # Compute total execution time from reported training data
    steps = report["step"]
    speed = report["speed"]
    interval = steps[1:] - steps[:-1]
    interval = np.hstack((interval[0], interval))
    interval_time = interval * speed

    report["time"] = np.cumsum(interval_time)
    total_time = report["time"][-1]

    # Find the start and end time for the power sampler
    wattage = power[:, 1:].sum(axis = 1)
    timestamps = power[:, 0]
    start_idx = np.where(wattage >= kernel_power)[0][0]
    start_idx = max(0, start_idx - 1)
    start_time = timestamps[start_idx]
    stop_time = start_time + total_time
    stop_idx = np.where(timestamps >= stop_time)[0][0]

    # Compute energy for the region of the sampled power where the training was active (not idle)
    joules = sp.integrate.cumtrapz(wattage[start_idx:stop_idx], (timestamps[start_idx:stop_idx] - start_time) / 3600.0, initial = 0.0)

    # Gather energy data for the number of steps reported
    idx = 0
    report["energy"] = []

    for steps_time in interval_time:
        stop_time = start_time + steps_time
        stop_idx = start_idx

        while True:
            if timestamps[stop_idx] >= stop_time:
                break

            stop_idx += 1
            idx += 1

        start_idx = stop_idx - 1 if start_idx != stop_idx - 1 else stop_idx
        idx -= 1

        report["energy"].append(joules[idx])
        start_time = timestamps[start_idx]

    report["energy"] = np.array(report["energy"])

    # %%

    """
    plt.figure(figsize = figsize)
    plt.title("Power data for the complete/unstopped training", fontsize = title_size)
    plt.xlabel("Execution time (hours)", fontsize = label_size)
    plt.ylabel("Power (Watts)", fontsize = label_size)

    plt.plot(power[:, 0] / 3600.0, power[:, 1:].sum(axis = 1), linewidth = line_width, label = "B = %d" % (batch_size))

    plt.grid()
    plt.xlim(xmin = 0.0, xmax = power[-1, 0] / 3600.0)
    plt.ylim(ymin = 0.0)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    plt.legend(fontsize = legend_size)
    plt.show()
    #"""

# %% Main program

if __name__ == "__main__":
    directory = "../Data"
    datasets = ["Wikipedia", "Childrens_Book_Test"]
    batch_sizes = [8, 16, 32, 64]
    loss_types = ["training", "validation"]

    powers = { dataset: { batch_size: get_power("%s/%s/batch_size_%d/power.dat" % (directory, dataset, batch_size)) for batch_size in batch_sizes } for dataset in datasets }
    reports = { dataset: { batch_size: get_report("%s/%s/batch_size_%d/log_main.json" % (directory, dataset, batch_size)) for batch_size in batch_sizes } for dataset in datasets }

    for dataset in datasets:
        for batch_size in batch_sizes:
            for loss_type in loss_types:
                power = powers[dataset][batch_size]
                report = reports[dataset][batch_size][loss_type]
                augment_data(power, report)

    # %%

    def time_vs_power(power, label = None, save_image = False):
        plt.figure(figsize = figsize)
        plt.title("Power data for the complete/unstopped training", fontsize = title_size)
        plt.xlabel("Execution time (hours)", fontsize = label_size)
        plt.ylabel("Power (Watts)", fontsize = label_size)

        plt.plot(power[:, 0] / 3600.0, power[:, 1:].sum(axis = 1), linewidth = line_width, label = label)

        plt.grid()
        plt.xlim(xmin = 0.0, xmax = power[-1, 0] / 3600.0)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/time_vs_power.pdf", bbox_inches = "tight")

    #"""
    save_image = False
    dataset = "Childrens_Book_Test"
    batch_size = 64

    time_vs_power(powers[dataset][batch_size], save_image = save_image)

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def time_vs_energy_converged(loss_type, accuracy_target, save_image = False, filename = "time_vs_energy_converged.pdf", filepath = "../Figures"):
        plt.figure(figsize = figsize)
        plt.title("Energy consumed to achieve a %s loss of '%1.2f'" % (loss_type, accuracy_target), fontsize = title_size)
        plt.xlabel("Execution time (hours)", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmin = np.inf
        xmax = 0.0
        ymax = 0.0

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                time = report["time"] / 3600.0
                loss = report["loss"]
                energy = report["energy"]
                target_idx = np.where(loss <= accuracy_target)[0][0]

                plt.plot(time[:target_idx], energy[:target_idx], color = color_cycle[jdx], linewidth = line_width, alpha = alpha if idx == 0 else 1.0, label = "B = %d" % (batch_size) if idx == 1 else None)
                plt.scatter(time[target_idx - 1], energy[target_idx - 1], color = color_cycle[jdx], s = marker_size, alpha = alpha if idx == 0 else 1.0)

                xmin = min(xmin, time[0])
                xmax = max(xmax, time[target_idx - 1])
                ymax = max(ymax, energy[target_idx - 1])

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.ylim(ymin = 0.0, ymax = ymax)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("%s/%s" % (filepath, filename), bbox_inches = "tight")

    #"""
    save_image = False
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/April_10_2023/Figures"

    accuracy_target = 1.9
    loss_type = "training"
    time_vs_energy_converged(loss_type, accuracy_target, save_image = save_image, filename = "time_vs_energy_converged_%s_%d.svg" % (loss_type, int(accuracy_target * 100.0)), filepath = filepath)

    accuracy_target = 2.0
    loss_type = "training"
    time_vs_energy_converged(loss_type, accuracy_target, save_image = save_image, filename = "time_vs_energy_converged_%s_%d.svg" % (loss_type, int(accuracy_target * 100.0)), filepath = filepath)

    accuracy_target = 3.0
    loss_type = "validation"
    time_vs_energy_converged(loss_type, accuracy_target, save_image = save_image, filename = "time_vs_energy_converged_%s_%d.svg" % (loss_type, int(accuracy_target * 100.0)), filepath = filepath)

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def accuracy_vs_energy(loss_type, save_image = False, filename = "accuracy_vs_energy.pdf", filepath = "../Figures"):
        index = { dataset: { batch_size: [] for batch_size in batch_sizes } for dataset in datasets }

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                loss = report["loss"]

                current_loss = np.inf

                for kdx in range(loss.shape[0]):
                    if loss[kdx] < current_loss:
                        current_loss = loss[kdx]
                        index[dataset][batch_size].append(kdx)

                index[dataset][batch_size] = np.array(index[dataset][batch_size], dtype = int)

        fig, ax = plt.subplots(figsize = figsize)
        plt.title("Energy as a function of %s loss" % (loss_type), fontsize = title_size)
        plt.xlabel("%s loss" % (loss_type.capitalize()), fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmin = np.inf
        xmax = 0.0

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                loss = report["loss"]
                energy = report["energy"]
                samples = index[dataset][batch_size]

                plt.semilogy(loss[samples], energy[samples], color = color_cycle[jdx], linewidth = line_width, alpha = alpha if idx == 0 else 1.0, label = "B = %d" % (batch_size) if idx == 1 else None)

                xmin = min(xmin, loss[samples[-1]])
                xmax = max(xmax, loss[samples[0]])

        plt.grid()
        plt.xlim(xmin = xmax, xmax = xmin)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        #if loss_type == "training":
        #    plt.legend(fontsize = legend_size, loc = 2)
        #else:
        #    plt.legend(fontsize = legend_size, loc = 4)

        #if loss_type == "training":
        #    axin = ax.inset_axes([0.55, 0.05, 0.43, 0.38])
        #else:
        #    axin = ax.inset_axes([0.22, 0.50, 0.43, 0.41])

        #for batch_size in batch_sizes:
        #    report = reports[batch_size][loss_type]
        #    axin.semilogy(accuracy[batch_size], joules[batch_size], linewidth = line_width, label = "B = %d" % (batch_size))

        #axin.grid()
        #axin.set_xlim(1.7, 2.2)
        #axin.set_ylim(8000.0, 20000.0)
        #axin.invert_xaxis()
        #axin.tick_params(axis = 'both', which = "major", labelsize = ticks_size * 0.7)
        #axin.tick_params(axis = 'both', which = "minor", labelsize = ticks_size * 0.7)
        #rect, line_1, line_2 = mark_inset(ax, axin, loc1 = 1, loc2 = 2, alpha = 0.4, zorder = 10)

        #if loss_type == "training":
        #    line_1.loc1 = 1
        #    line_1.loc2 = 3
        #    line_2.loc1 = 2
        #    line_2.loc2 = 4
        #else:
        #    line_1.loc1 = 1
        #    line_1.loc2 = 1
        #    line_2.loc1 = 4
        #    line_2.loc2 = 4

        if save_image:
            plt.savefig("%s/%s" % (filepath, filename), bbox_inches = "tight")

    #"""
    save_image = False
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/April_10_2023/Figures"

    loss_type = "training"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_energy(loss_type, save_image = save_image, filename = filename, filepath = filepath)

    loss_type = "validation"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_energy(loss_type, save_image = save_image, filename = filename, filepath = filepath)

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def accuracy_vs_time(loss_type, save_image = False, filename = "accuracy_vs_energy.pdf", filepath = "../Figures"):
        index = { dataset: { batch_size: [] for batch_size in batch_sizes } for dataset in datasets }

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                loss = report["loss"]

                current_loss = np.inf

                for kdx in range(loss.shape[0]):
                    if loss[kdx] < current_loss:
                        current_loss = loss[kdx]
                        index[dataset][batch_size].append(kdx)

                index[dataset][batch_size] = np.array(index[dataset][batch_size], dtype = int)

        fig, ax = plt.subplots(figsize = figsize)
        plt.title("Total execution time as a function of %s loss" % (loss_type), fontsize = title_size)
        plt.xlabel("%s loss" % (loss_type.capitalize()), fontsize = label_size)
        plt.ylabel("Execution time (hours)", fontsize = label_size)

        xmin = np.inf
        xmax = 0.0

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                loss = report["loss"]
                time = report["time"] / 3600.0
                samples = index[dataset][batch_size]

                plt.semilogy(loss[samples], time[samples], color = color_cycle[jdx], linewidth = line_width, alpha = alpha if idx == 0 else 1.0, label = "B = %d" % (batch_size) if idx == 1 else None)

                xmin = min(xmin, loss[samples[-1]])
                xmax = max(xmax, loss[samples[0]])

        plt.grid()
        plt.xlim(xmin = xmax, xmax = xmin)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        #if loss_type == "training":
        #    plt.legend(fontsize = legend_size, loc = 2)
        #else:
        #    plt.legend(fontsize = legend_size, loc = 4)

        #if loss_type == "training":
        #    axin = ax.inset_axes([0.55, 0.05, 0.43, 0.38])
        #else:
        #    axin = ax.inset_axes([0.22, 0.50, 0.43, 0.41])

        #for batch_size in batch_sizes:
        #    report = reports[batch_size][loss_type]
        #    axin.semilogy(accuracy[batch_size], joules[batch_size], linewidth = line_width, label = "B = %d" % (batch_size))

        #axin.grid()
        #axin.set_xlim(1.7, 2.2)
        #axin.set_ylim(8000.0, 20000.0)
        #axin.invert_xaxis()
        #axin.tick_params(axis = 'both', which = "major", labelsize = ticks_size * 0.7)
        #axin.tick_params(axis = 'both', which = "minor", labelsize = ticks_size * 0.7)
        #rect, line_1, line_2 = mark_inset(ax, axin, loc1 = 1, loc2 = 2, alpha = 0.4, zorder = 10)

        #if loss_type == "training":
        #    line_1.loc1 = 1
        #    line_1.loc2 = 3
        #    line_2.loc1 = 2
        #    line_2.loc2 = 4
        #else:
        #    line_1.loc1 = 1
        #    line_1.loc2 = 1
        #    line_2.loc1 = 4
        #    line_2.loc2 = 4

        if save_image:
            plt.savefig("%s/%s" % (filepath, filename), bbox_inches = "tight")

    #"""
    save_image = False
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/April_10_2023/Figures"

    loss_type = "training"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_time(loss_type, save_image = save_image, filename = filename, filepath = filepath)

    loss_type = "validation"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_time(loss_type, save_image = save_image, filename = filename, filepath = filepath)

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def time_vs_accuracy(report_type, save_image = False, filename = "time_vs_accuracy.pdf"):
        plt.figure(figsize = figsize)
        plt.title("Convergence of the training as function of the execution time", fontsize = title_size)
        plt.xlabel("Execution time (hours)", fontsize = label_size)
        plt.ylabel("Training loss", fontsize = label_size)

        xmin = np.inf
        xmax = 0.0

        for batch_size in batch_sizes:
            report = reports[batch_size][report_type]
            steps = report["step"]
            loss = report["loss"]
            speed = report["speed"]

            interval = steps[1:] - steps[:-1]
            interval = np.hstack((interval[0], interval))
            total_time = np.cumsum(interval * speed)

            plt.plot(total_time / 3600.0, loss, linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, total_time.min() / 3600.0)
            xmax = max(xmax, total_time.max() / 3600.0)

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    """
    report_type = "training"
    time_vs_accuracy(report_type, save_image = True, filename = "time_vs_accuracy_%s.pdf" % (report_type))

    report_type = "validation"
    time_vs_accuracy(report_type, save_image = True, filename = "time_vs_accuracy_%s.pdf" % (report_type))

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def time_vs_energy(loss_type, save_image = False, filename = "time_vs_energy.pdf"):
        plt.figure(figsize = figsize)
        plt.title("Energy used as a function of the execution time", fontsize = title_size)
        plt.xlabel("Execution time (hours)", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmin = np.inf
        xmax = np.inf

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                time = report["time"] / 3600.0
                energy = report["energy"]

                plt.plot(time, energy, color = color_cycle[jdx], linewidth = line_width, alpha = alpha if idx == 0 else 1.0, label = "B = %d" % (batch_size) if idx == 1 else None)

                if time[0] < xmin:
                    xmin = time[0]
                    ymin = energy[0]

                if time[-1] < xmax:
                    xmax = time[-1]
                    ymax = energy[-1]

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.ylim(ymin = 0.0, ymax = ymax)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    #"""
    loss_type = "training"
    time_vs_energy(loss_type, save_image = True, filename = "time_vs_energy_%s.pdf" % (loss_type))

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def convergence_history(loss_type, save_image = False, filename = "convergence_history.pdf"):
        fig, ax = plt.subplots(figsize = figsize)
        plt.title("Convergence of the %s loss as a function of the number of steps" % (loss_type), fontsize = title_size)
        plt.xlabel("Number of steps (thousands)", fontsize = label_size)
        plt.ylabel("%s loss" % (loss_type.capitalize()), fontsize = label_size)

        xmin = np.inf
        xmax = np.inf

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                steps = report["step"] / 1000.0
                loss = report["loss"]

                plt.plot(steps, loss, color = color_cycle[jdx], linewidth = line_width, alpha = alpha if idx == 0 else 1.0, label = "B = %d" % (batch_size) if idx == 1 else None)

                xmin = min(xmin, steps[0])
                xmax = min(xmax, steps[-1])

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    #"""
    save_image = False

    loss_type = "training"
    convergence_history(loss_type, save_image = save_image, filename = "convergence_history_%s.pdf" % (loss_type))

    loss_type = "validation"
    convergence_history(loss_type, save_image = save_image, filename = "convergence_history_%s.pdf" % (loss_type))

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def energy_history(loss_type, save_image = False, filename = "energy_history.pdf"):
        plt.figure(figsize = figsize)
        plt.title("Energy used as a function of the number of steps", fontsize = title_size)
        plt.xlabel("Number of steps (thousands)", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmin = np.inf
        xmax = np.inf

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                steps = report["step"] / 1000.0
                energy = report["energy"]

                plt.plot(steps, energy, color = color_cycle[jdx], linewidth = line_width, alpha = alpha if idx == 0 else 1.0, label = "B = %d" % (batch_size) if idx == 1 else None)

                if steps[0] < xmin:
                    xmin = steps[0]
                    ymin = energy[0]

                if steps[-1] < xmax:
                    xmax = steps[-1]
                    ymax = energy[-1]

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.ylim(ymin = 0.0, ymax = ymax)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    #"""
    save_image = False

    loss_type = "training"
    energy_history(loss_type, save_image = save_image, filename = "energy_history_%s.pdf" % (loss_type))

    loss_type = "validation"
    energy_history(loss_type, save_image = save_image, filename = "energy_history_%s.pdf" % (loss_type))

    if show_plots_individually:
        plt.show()
    #"""

    # %%

    def training_speed_history(loss_type, save_image = False, filename = "training_speed_history.pdf"):
        plt.figure(figsize = figsize)
        plt.title("Training speed as a function of the number of steps", fontsize = title_size)
        plt.xlabel("Number of steps (thousands)", fontsize = label_size)
        plt.ylabel("Training speed", fontsize = label_size)

        xmin = np.inf
        xmax = np.inf

        for idx, dataset in enumerate(datasets):
            for jdx, batch_size in enumerate(batch_sizes):
                report = reports[dataset][batch_size][loss_type]
                steps = report["step"] / 1000.0
                speeds = report["speed"]

                plt.plot(steps, speeds, color = color_cycle[jdx], linewidth = line_width, alpha = alpha if idx == 0 else 1.0, label = "B = %d" % (batch_size) if idx == 1 else None)

                xmin = min(xmin, steps[0])
                xmax = min(xmax, steps[-1])

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    #"""
    loss_type = "training"
    training_speed_history(loss_type)

    loss_type = "validation"
    training_speed_history(loss_type)

    if show_plots_individually:
        plt.show()
    #"""

# %% Plotting

    """
    save_image = False
    dataset = "Childrens_Book_Test"
    batch_size = 64

    time_vs_power(powers[dataset][batch_size], save_image = save_image)
    #"""

    #"""
    save_image = False
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/April_10_2023/Figures"

    accuracy_target = 1.9
    loss_type = "training"
    time_vs_energy_converged(loss_type, accuracy_target, save_image = save_image, filename = "time_vs_energy_converged_%s_%d.svg" % (loss_type, int(accuracy_target * 100.0)), filepath = filepath)

    accuracy_target = 2.0
    loss_type = "training"
    time_vs_energy_converged(loss_type, accuracy_target, save_image = save_image, filename = "time_vs_energy_converged_%s_%d.svg" % (loss_type, int(accuracy_target * 100.0)), filepath = filepath)

    accuracy_target = 3.0
    loss_type = "validation"
    time_vs_energy_converged(loss_type, accuracy_target, save_image = save_image, filename = "time_vs_energy_converged_%s_%d.svg" % (loss_type, int(accuracy_target * 100.0)), filepath = filepath)
    #"""

    #"""
    save_image = False
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/April_10_2023/Figures"

    loss_type = "training"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_energy(loss_type, save_image = save_image, filename = filename, filepath = filepath)

    loss_type = "validation"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_energy(loss_type, save_image = save_image, filename = filename, filepath = filepath)
    #"""

    #"""
    save_image = False
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/April_10_2023/Figures"

    loss_type = "training"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_time(loss_type, save_image = save_image, filename = filename, filepath = filepath)

    loss_type = "validation"
    filename = "accuracy_vs_energy_%s.svg" % (loss_type)
    accuracy_vs_time(loss_type, save_image = save_image, filename = filename, filepath = filepath)
    #"""

    #"""
    loss_type = "training"
    time_vs_energy(loss_type, save_image = True, filename = "time_vs_energy_%s.pdf" % (loss_type))
    #"""

    #"""
    save_image = False

    loss_type = "training"
    convergence_history(loss_type, save_image = save_image, filename = "convergence_history_%s.pdf" % (loss_type))

    loss_type = "validation"
    convergence_history(loss_type, save_image = save_image, filename = "convergence_history_%s.pdf" % (loss_type))
    #"""

    #"""
    save_image = False

    loss_type = "training"
    energy_history(loss_type, save_image = save_image, filename = "energy_history_%s.pdf" % (loss_type))

    loss_type = "validation"
    energy_history(loss_type, save_image = save_image, filename = "energy_history_%s.pdf" % (loss_type))
    #"""

    #"""
    loss_type = "training"
    training_speed_history(loss_type)

    loss_type = "validation"
    training_speed_history(loss_type)
    #"""

    plt.show()

# %% End of program
