"""
SusQL demo plotting
"""

# %% Modules

"""
import matplotlib

matplotlib.use("Qt5agg")
"""

import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt

from time import sleep
from datetime import datetime

# %% Formating

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
line_width = 2.0
title_size = 19
label_size = 17
legend_size = 18
ticks_size = 15

# %% Functions

# %% Main program

def main():
    """
    Main function
    """

    # %%

    IS_MONITORING = True

    directory = "/home/metalcycling/Documents/Sustainability/Code/SusQL/vault-2"
    #directory = "/home/metalcycling/Documents/Sustainability/Code/SusQL/vault"
    vault_files = glob.glob("%s/*.yaml" % (directory))

    fig, ax = plt.subplots(figsize = (12, 8))
    plt.title("SusQL energy measurements", fontsize = title_size)
    plt.xlabel("Time (seconds)", fontsize = label_size)
    plt.ylabel("Energy (Joules)", fontsize = label_size)
    plt.grid()
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)

    for vault_file in vault_files:
        filename = vault_file.split("/")[-1]
        plt.plot([], [], label = filename.split(".")[0])

    t_min = np.inf
    t_max = 0.0
    e_min = np.inf
    e_max = 0.0

    while IS_MONITORING:
        if len(vault_files) == 0:
            continue

        for fdx, vault_file in enumerate(vault_files):
            filename = vault_file.split("/")[-1]

            try:
                measurements = yaml.safe_load(open(vault_file, "r"))
            except:
                continue

            if isinstance(measurements, type(None)):
                continue

            if "timestamp" in measurements:
                timestamps = measurements["timestamp"]
            else:
                continue

            if "total_energy" in measurements:
                total_energy = measurements["total_energy"]
            else:
                continue

            if len(timestamps) != len(total_energy):
                continue

            execution_time = [0.0]

            for tdx in range(len(timestamps) - 1):
                t_start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
                t_stop = datetime.strptime(timestamps[tdx + 1], "%Y-%m-%d %H:%M:%S")

                execution_time.append((t_stop - t_start).total_seconds())

            ax.lines[fdx].set_xdata(execution_time)
            ax.lines[fdx].set_ydata(total_energy)

            t_min = execution_time[0]
            t_max = execution_time[-1]
            e_min = min(e_min, total_energy[0])
            e_max = max(e_max, total_energy[-1])

        plt.xlim(xmin = t_min, xmax = t_max)
        plt.ylim(ymin = e_min , ymax = e_max * 1.05)
        plt.legend(fontsize = legend_size)

        plt.draw()
        plt.pause(0.05)

        break

    # %%

    #directory = "/home/metalcycling/Documents/Sustainability/Code/SusQL/vault-2"
    directory = "/home/metalcycling/Documents/Sustainability/Code/SusQL/vault"
    vault_files = glob.glob("%s/*.yaml" % (directory))

    plt.figure(figsize = (12, 8))
    plt.title("SusQL energy measurements", fontsize = title_size)
    plt.xlabel("Time (seconds)", fontsize = label_size)
    plt.ylabel("Energy (Joules)", fontsize = label_size)

    t_min = np.inf
    t_max = 0.0

    for vault_file in vault_files:
        filename = vault_file.split("/")[-1]
        measurements = yaml.safe_load(open(vault_file, "r"))
        timestamps = measurements["timestamp"]
        total_energy = measurements["total_energy"]

        execution_time = [0.0]

        for tdx in range(len(timestamps) - 1):
            t_start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            t_stop = datetime.strptime(timestamps[tdx + 1], "%Y-%m-%d %H:%M:%S")

            execution_time.append((t_stop - t_start).total_seconds())

        plt.plot(execution_time, total_energy, "-", linewidth = line_width, label = filename.split(".")[0])

        t_min = execution_time[0]
        t_max = execution_time[-1]

    plt.grid()
    plt.xlim(xmin = t_min, xmax = t_max)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    plt.legend(fontsize = legend_size)
    plt.show()

# %% Main program

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            quit()
        except SystemExit:
            quit()

# %% End of program
