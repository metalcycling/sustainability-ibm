"""
Pareto front computation

%load_ext autoreload
%autoreload 2
"""

# %% Modules

import json
import numpy as np
import scipy as sp

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

def accuracy_vs_time(report_type):
    accuracy = { batch_size: [] for batch_size in batch_sizes }
    execution_time = { batch_size: [] for batch_size in batch_sizes }
    index = { batch_size: [] for batch_size in batch_sizes }

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

        accuracy[batch_size] = np.array(accuracy[batch_size])
        execution_time[batch_size] = np.array(execution_time[batch_size])
    
    return accuracy, execution_time

def intersection(curve_1, curve_2):
    func_1 = sp.interpolate.interp1d(curve_1["x"], curve_1["y"], kind = "linear")
    func_2 = sp.interpolate.interp1d(curve_2["x"], curve_2["y"], kind = "linear")

    x = np.linspace(max(curve_1["x"].min(), curve_2["x"].min()), min(curve_1["x"].max(), curve_2["x"].max()), max(curve_1["x"].shape[0], curve_2["x"].shape[0]))

    y_1 = func_1(x)
    y_2 = func_2(x)

    idx = np.argwhere(np.diff(np.sign(y_1 - y_2))).flatten()

    return x[idx[-1]], y_1[idx[-1]]

    # %% Parse data

directory = "../Data"
batch_sizes = [8, 16, 32, 64]
powers = { batch_size: get_power("%s/batch_size_%d/power.dat" % (directory, batch_size)) for batch_size in batch_sizes }
reports = { batch_size: get_report("%s/batch_size_%d/log_main.json" % (directory, batch_size)) for batch_size in batch_sizes }
time_in_hours = True

# %% Build graphs

report_type = "training"
accuracy, execution_time = accuracy_vs_time(report_type)

#execution_time[16] - execution_time[8]

curve_1 = { "x": accuracy[8], "y": execution_time[8] }
curve_2 = { "x": accuracy[16], "y": execution_time[16] }
accuracy_inter, execution_time_inter = intersection(curve_1, curve_2)

# %% End of program
