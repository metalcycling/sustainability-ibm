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

if presentation_mode:
    figsize = (26, 18)
    title_size = 26
    label_size = 24
    ticks_size = 24
    legend_size = 21
    line_width = 3
    marker_size = 44

else:
    figsize = (12, 8)
    title_size = 22
    label_size = 20
    ticks_size = 20
    legend_size = 17
    line_width = 3
    marker_size = 40

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

# %% Main program

if __name__ == "__main__":
    directory = "../Data"
    batch_sizes = [8, 16, 32, 64]
    powers = { batch_size: get_power("%s/batch_size_%d/power.dat" % (directory, batch_size)) for batch_size in batch_sizes }
    reports = { batch_size: get_report("%s/batch_size_%d/log_main.json" % (directory, batch_size)) for batch_size in batch_sizes }
    time_in_hours = True

    # %%

    def time_vs_power(save_image = False):
        plt.figure(figsize = figsize)
        plt.title("Power data for the complete/unstopped training", fontsize = title_size)
        plt.xlabel("Execution time (hours)" if time_in_hours else "Execution time (seconds)", fontsize = label_size)
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

        if save_image:
            plt.savefig("../Figures/time_vs_power.pdf", bbox_inches = "tight")

    """
    time_vs_power()
    plt.show()
    #"""

    # %%

    def time_vs_energy_converged(report_type, threshold, idle_power, save_image = False, filename = "time_vs_energy_converged.pdf", filepath = "../Figures"):
        plt.figure(figsize = figsize)
        plt.title("Energy consumed to achieve a %s loss of '%1.2f'" % (report_type, threshold), fontsize = title_size)
        plt.xlabel("Execution time (hours)" if time_in_hours else "Execution time (seconds)", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmax = 0.0

        for bdx, batch_size in enumerate(batch_sizes):
            power = powers[batch_size]
            report = reports[batch_size][report_type]
            timestamps = power[:, 0]
            wattage = power[:, 1:].sum(axis = 1)

            # Get completion time based on the threshold
            stop_idx = np.where(report["loss"] < threshold)[0][0]
            steps = report["step"][:stop_idx]
            speed = report["speed"][:stop_idx]
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
                plt.scatter((timestamps[stop_idx - 1] - start_time) / 3600.0, joules[-1], color = color_cycle[bdx], s = marker_size)
            else:
                plt.plot(timestamps[start_idx:stop_idx] - start_time, joules, linewidth = line_width, label = "B = %d" % (batch_size))
                plt.scatter(timestamps[stop_idx - 1] - start_time, joules[-1], color = color_cycle[bdx], s = marker_size)

            xmax = max(xmax, stop_time / 3600.0 if time_in_hours else stop_time)

        plt.grid()
        plt.xlim(xmin = 0.0, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("%s/%s" % (filepath, filename), bbox_inches = "tight")

    """
    idle_power = 560.0

    threshold = 1.9
    report_type = "training"
    time_vs_energy_converged(report_type, threshold, idle_power, save_image = True, filename = "time_vs_energy_converged_%s_%d.pdf" % (report_type, int(threshold * 100.0)))

    threshold = 2.1
    report_type = "training"
    time_vs_energy_converged(report_type, threshold, idle_power, save_image = True, filename = "time_vs_energy_converged_%s_%d.pdf" % (report_type, int(threshold * 100.0)))

    threshold = 1.85
    report_type = "validation"
    time_vs_energy_converged(report_type, threshold, idle_power, save_image = True, filename = "time_vs_energy_converged_%s_%d.pdf" % (report_type, int(threshold * 100.0)))

    threshold = 2.0
    report_type = "validation"
    time_vs_energy_converged(report_type, threshold, idle_power, save_image = True, filename = "time_vs_energy_converged_%s_%d.pdf" % (report_type, int(threshold * 100.0)))

    plt.show()
    #"""

    #"""
    idle_power = 560.0

    threshold = 1.9
    report_type = "training"
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/March_30_2023/Figures"
    filename = "time_vs_energy_converged_%s_%d.svg" % (report_type, int(threshold * 100.0))
    time_vs_energy_converged(report_type, threshold, idle_power, save_image = True, filename = filename, filepath = filepath)

    plt.show()
    #"""

    # %%

    def accuracy_vs_energy(report_type, idle_power, save_image = False, filename = "accuracy_vs_energy.pdf", filepath = "../Figures"):
        xmin = np.inf
        xmax = 0.0

        figsize = (20, 7)
        title_size = 30
        label_size = 28
        ticks_size = 28
        legend_size = 25
        line_width = 5
        marker_size = 40


        fig, ax = plt.subplots(figsize = figsize)
        plt.title("Energy as a function of %s loss" % (report_type), fontsize = title_size)
        plt.xlabel("%s loss" % (report_type.capitalize()), fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        accuracy = { batch_size: [] for batch_size in batch_sizes }
        joules = { batch_size: [] for batch_size in batch_sizes }
        index = { batch_size: [] for batch_size in batch_sizes }

        for batch_size in batch_sizes:
            power = powers[batch_size]
            report = reports[batch_size][report_type]
            timestamps = power[:, 0]
            loss = report["loss"]
            steps = report["step"]
            speed = report["speed"]
            wattage = power[:, 1:].sum(axis = 1)

            interval = steps[1:] - steps[:-1]
            interval = np.hstack((interval[0], interval))
            total_time = np.cumsum(interval * speed)

            start_idx = np.where(wattage >= idle_power)[0][0]
            start_time = timestamps[start_idx]

            current_loss = np.inf

            for jdx in range(loss.shape[0]):
                if loss[jdx] < current_loss:
                    current_loss = loss[jdx]
                    stop_time = start_time + total_time[jdx]
                    stop_idx = np.where(timestamps > stop_time)[0][0]

                    index[batch_size].append(jdx)
                    accuracy[batch_size].append(current_loss)
                    joules[batch_size].append(sp.integrate.trapz(wattage[start_idx:stop_idx], (timestamps[start_idx:stop_idx] - start_time) / 3600.0))

            plt.semilogy(accuracy[batch_size], joules[batch_size], linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, accuracy[batch_size][-1])
            xmax = max(xmax, accuracy[batch_size][0])

        plt.grid()
        plt.xlim(xmin = xmax, xmax = xmin)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)

        if report_type == "training":
            plt.legend(fontsize = legend_size, loc = 2)
        else:
            plt.legend(fontsize = legend_size, loc = 4)

        if report_type == "training":
            #axin = ax.inset_axes([0.55, 0.05, 0.43, 0.38])
            axin = ax.inset_axes([0.55, 0.1, 0.43, 0.38])
        else:
            axin = ax.inset_axes([0.22, 0.50, 0.43, 0.41])

        for batch_size in batch_sizes:
            report = reports[batch_size][report_type]
            axin.semilogy(accuracy[batch_size], joules[batch_size], linewidth = line_width, label = "B = %d" % (batch_size))

        axin.grid()
        axin.set_xlim(1.7, 2.2)
        axin.set_ylim(8000.0, 20000.0)
        axin.invert_xaxis()
        axin.tick_params(axis = 'both', which = "major", labelsize = ticks_size * 0.7)
        axin.tick_params(axis = 'both', which = "minor", labelsize = ticks_size * 0.7)
        rect, line_1, line_2 = mark_inset(ax, axin, loc1 = 1, loc2 = 2, alpha = 0.4, zorder = 10)

        if report_type == "training":
            line_1.loc1 = 1
            line_1.loc2 = 3
            line_2.loc1 = 2
            line_2.loc2 = 4
        else:
            line_1.loc1 = 1
            line_1.loc2 = 1
            line_2.loc1 = 4
            line_2.loc2 = 4

        if save_image:
            plt.savefig("%s/%s" % (filepath, filename), bbox_inches = "tight")

    """
    idle_power = 560.0

    report_type = "training"
    accuracy_vs_energy(report_type, idle_power, save_image = True, filename = "accuracy_vs_energy_%s.pdf" % (report_type))

    report_type = "validation"
    accuracy_vs_energy(report_type, idle_power, save_image = True, filename = "accuracy_vs_energy_%s.pdf" % (report_type))

    plt.show()
    #"""

    #"""
    idle_power = 560.0

    threshold = 1.9
    report_type = "training"
    save_image = True
    #filepath = "/home/metalcycling/Documents/Sustainability/Presentations/March_30_2023/Figures"
    filepath = "/home/metalcycling"
    filename = "accuracy_vs_energy_%s.svg" % (report_type)
    accuracy_vs_energy(report_type, idle_power, save_image = save_image, filename = filename, filepath = filepath)

    plt.show()
    #"""

    # %%

    def accuracy_vs_time(report_type, save_image = False, filename = "accuracy_vs_time.pdf", filepath = "../Figures"):
        fig, ax = plt.subplots(figsize = figsize)
        plt.title("Total execution time as a function of %s loss" % (report_type), fontsize = title_size)
        plt.xlabel("%s loss" % (report_type.capitalize()), fontsize = label_size)
        plt.ylabel("Execution time (hours)", fontsize = label_size)

        accuracy = { batch_size: [] for batch_size in batch_sizes }
        execution_time = { batch_size: [] for batch_size in batch_sizes }
        index = { batch_size: [] for batch_size in batch_sizes }

        xmin = np.inf
        xmax = 0.0

        for batch_size in batch_sizes:

            report = reports[batch_size][report_type]
            speed = report["speed"]
            loss = report["loss"]
            steps = report["step"]

            interval = steps[1:] - steps[:-1]
            interval = np.hstack((interval[0], interval))
            total_time = np.cumsum(interval * speed)

            current_loss = np.inf

            for jdx in range(loss.shape[0]):
                if loss[jdx] < current_loss:
                    current_loss = loss[jdx]

                    index[batch_size].append(jdx)
                    accuracy[batch_size].append(current_loss)
                    execution_time[batch_size].append(total_time[jdx] / 3600.0)

            plt.semilogy(accuracy[batch_size], execution_time[batch_size], linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, accuracy[batch_size][-1])
            xmax = max(xmax, accuracy[batch_size][0])

        plt.grid()
        plt.xlim(xmin = xmax, xmax = xmin)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)

        if report_type == "training":
            plt.legend(fontsize = legend_size, loc = 2)
        else:
            plt.legend(fontsize = legend_size, loc = 4)

        if report_type == "training":
            axin = ax.inset_axes([0.55, 0.05, 0.43, 0.38])
        else:
            axin = ax.inset_axes([0.22, 0.50, 0.43, 0.41])

        for batch_size in batch_sizes:
            report = reports[batch_size][report_type]
            axin.semilogy(accuracy[batch_size], execution_time[batch_size], linewidth = line_width, label = "B = %d" % (batch_size))

        axin.grid()
        axin.set_xlim(1.7, 2.2)
        axin.set_ylim(3.0, 10.0)
        axin.tick_params(axis = 'both', which = "major", labelsize = ticks_size * 0.7)
        axin.tick_params(axis = 'both', which = "minor", labelsize = ticks_size * 0.7)
        axin.invert_xaxis()
        rect, line_1, line_2 = mark_inset(ax, axin, loc1 = 1, loc2 = 2, alpha = 0.4, zorder = 10)

        if report_type == "training":
            line_1.loc1 = 1
            line_1.loc2 = 3
            line_2.loc1 = 2
            line_2.loc2 = 4
        else:
            line_1.loc1 = 1
            line_1.loc2 = 1
            line_2.loc1 = 4
            line_2.loc2 = 4

        if save_image:
            plt.savefig("%s/%s" % (filepath, filename), bbox_inches = "tight")

    """
    report_type = "training"
    accuracy_vs_time(report_type, save_image = True, filename = "accuracy_vs_time_%s.pdf" % (report_type))

    report_type = "validation"
    accuracy_vs_time(report_type, save_image = True, filename = "accuracy_vs_time_%s.pdf" % (report_type))

    plt.show()
    #"""

    #"""
    save_image = False
    report_type = "training"
    filepath = "/home/metalcycling/Documents/Sustainability/Presentations/March_30_2023/Figures"
    filename = "accuracy_vs_time_%s.svg" % (report_type)
    accuracy_vs_time(report_type, save_image = save_image, filename = filename, filepath = filepath)

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

    plt.show()
    #"""

    # %%

    def time_vs_energy(report_type, idle_power, save_image = False, filename = "time_vs_energy.pdf"):
        plt.figure(figsize = figsize)
        plt.title("Energy used as a function of the execution time", fontsize = title_size)
        plt.xlabel("Execution time (hours)" if time_in_hours else "Execution time (seconds)", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

        xmin = np.inf
        xmax = 0.0
        joules = { batch_size: [] for batch_size in batch_sizes }

        for batch_size in batch_sizes:
            report = reports[batch_size][report_type]
            speed = report["speed"]
            steps = report["step"]

            interval = steps[1:] - steps[:-1]
            interval = np.hstack((interval[0], interval))
            total_time = np.cumsum(interval * speed)

            power = powers[batch_size]
            wattage = power[:, 1:].sum(axis = 1)
            timestamps = power[:, 0]
            start_idx = np.where(wattage >= idle_power)[0][0]
            start_time = timestamps[start_idx]

            jdx = 0

            for idx in range(start_idx, timestamps.shape[0]):
                if timestamps[idx] - start_time > total_time[jdx]:
                    stop_idx = idx
                    joules[batch_size].append(sp.integrate.trapz(wattage[start_idx:stop_idx], (timestamps[start_idx:stop_idx] - start_time) / 3600.0))
                    jdx += 1

                if jdx == len(total_time):
                    break

            plt.plot(total_time / 3600.0, joules[batch_size], linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, total_time[0] / 3600.0)
            xmax = max(xmax, total_time[-1] / 3600.0)

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    """
    idle_power = 560.0

    report_type = "training"
    time_vs_energy(report_type, idle_power, save_image = True, filename = "time_vs_energy_%s.pdf" % (report_type))

    report_type = "validation"
    time_vs_energy(report_type, idle_power, save_image = True, filename = "time_vs_energy_%s.pdf" % (report_type))

    plt.show()
    #"""

    # %%

    def convergence_history(report_type, save_image = False, filename = "convergence_history.pdf"):
        fig, ax = plt.subplots(figsize = figsize)
        plt.title("Convergence of the %s loss as a function of the number of steps" % (report_type), fontsize = title_size)
        plt.xlabel("Thousand steps", fontsize = label_size)
        plt.ylabel("%s loss" % (report_type.capitalize()), fontsize = label_size)

        xmin = np.inf
        xmax = 0.0

        for batch_size in batch_sizes:
            training = reports[batch_size][report_type]
            plt.plot(training["step"] / 1000.0, training["loss"], linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, training["step"][0] / 1000.0)
            xmax = max(xmax, training["step"][-1] / 1000.0)

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

    report_type = "training"
    convergence_history(report_type, save_image = save_image, filename = "convergence_history_%s.pdf" % (report_type))

    report_type = "validation"
    convergence_history(report_type, save_image = save_image, filename = "convergence_history_%s.pdf" % (report_type))

    plt.show()
    #"""

    # %%

    def energy_history(report_type, idle_power, save_image = False, filename = "energy_history.pdf"):
        plt.figure(figsize = figsize)
        plt.title("Energy used as a function of the number of steps", fontsize = title_size)
        plt.xlabel("Thousand steps", fontsize = label_size)
        plt.ylabel("Energy (Joules)", fontsize = label_size)

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

            power = powers[batch_size]
            wattage = power[:, 1:].sum(axis = 1)
            timestamps = power[:, 0]
            start_idx = np.where(wattage >= idle_power)[0][0]
            start_time = timestamps[start_idx]

            joules = []
            jdx = 0

            for idx in range(start_idx, timestamps.shape[0]):
                if timestamps[idx] - start_time > total_time[jdx]:
                    stop_idx = idx
                    joules.append(sp.integrate.trapz(wattage[start_idx:stop_idx], (timestamps[start_idx:stop_idx] - start_time) / 3600.0))
                    jdx += 1

                if jdx == len(total_time):
                    break

            plt.plot(steps / 1000.0, joules, linewidth = line_width, label = "B = %d" % (batch_size))

            xmin = min(xmin, steps[0] / 1000.0)
            xmax = max(xmax, steps[-1] / 1000.0)

        plt.grid()
        plt.xlim(xmin = xmin, xmax = xmax)
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    """
    idle_power = 560.0

    report_type = "training"
    energy_history(report_type, idle_power, save_image = True, filename = "energy_history_%s.pdf" % (report_type))

    report_type = "validation"
    energy_history(report_type, idle_power, save_image = True, filename = "energy_history_%s.pdf" % (report_type))

    plt.show()
    #"""

    # %%

    def training_speed_history(report_type, save_image = False, filename = "training_speed_history.pdf"):
        plt.figure(figsize = figsize)
        plt.title("Training speed as a function of the number of steps", fontsize = title_size)
        plt.xlabel("Step", fontsize = label_size)
        plt.ylabel("Training speed", fontsize = label_size)

        for batch_size in batch_sizes:
            report = reports[batch_size][report_type]
            plt.plot(report["step"], report["speed"], linewidth = line_width, label = "B = %d" % (batch_size))

        plt.grid()
        plt.xlim(xmin = 10.0, xmax = max([report["step"][-1] for batch_size in batch_sizes]))
        plt.ylim(ymin = 0.0)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.legend(fontsize = legend_size)

        if save_image:
            plt.savefig("../Figures/%s" % (filename), bbox_inches = "tight")

    """
    report_type = "training"
    training_speed_history(report_type)

    plt.show()
    #"""

# %% Plotting

    batch_size = 64
    report = reports[batch_size][report_type]
    speed = report["speed"]
    step = report["step"]

    interval_delta = 10
    num_steps = 1000
    stop_idx = num_steps // interval_delta

    #print(np.sum(speed[:stop_idx] * interval_delta) / 60.0)
    print(np.sum(speed[:stop_idx] * interval_delta) / 3600.0)

# %% Plotting
    
    threshold = 2.0
    idle_power = 560.0

    #time_vs_power(save_image = True)
    #time_vs_accuracy(save_image = True)
    #time_vs_energy(idle_power, save_image = True)
    #time_vs_energy_converged(threshold, idle_power, save_image = True, filename = "time_vs_energy_converged_20.pdf")
    #convergence_history(save_image = True)
    #energy_history(idle_power, save_image = True)
    #training_speed_history(save_image = True)

#    time_vs_energy_converged(2.1, idle_power, save_image = True, filename = "time_vs_energy_converged_21.pdf")
#    time_vs_energy_converged(1.9, idle_power, save_image = True, filename = "time_vs_energy_converged_19.pdf")

    accuracy_vs_energy(idle_power, save_image = True)
    accuracy_vs_time(save_image = True)

    plt.show()

# %% End of program
