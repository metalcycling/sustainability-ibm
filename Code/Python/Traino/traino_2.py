# %% Image classification
import time
import glob
import pickle
import pandas
import sklearn
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
from torch.utils.data import Dataset

# %% Runtime parameters
device = torch.device("cuda:0")
#device = torch.device("cpu")
#data_dir = "/home/metalcycling/Documents/Sustainability/Data/ImageNet/8x8"
data_dir = "/home/metalcycling/Documents/Sustainability/Data/ImageNet/16x16"
#data_dir = "/home/metalcycling/Documents/Sustainability/Data/Iris"

# %% Utilities
class MeterType(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class Meter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt = ":f", meter_type = MeterType.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.meter_type = meter_type
        self.reset()

    def reset(self):
        self.count = 0
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val, n = 1):
        self.count += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""

        if self.meter_type is MeterType.NONE:
            fmtstr = ""

        elif self.meter_type is MeterType.AVERAGE:
            fmtstr = "{name} {avg:.3f}"

        elif self.meter_type is MeterType.SUM:
            fmtstr = "{name} {sum:.3f}"

        elif self.meter_type is MeterType.COUNT:
            fmtstr = "{name} {count:.3f}"

        else:
            raise ValueError("invalid meter type %r" % self.meter_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """
    Displays a progress for a set of meters
    """
    def __init__(self, num_batches, meters, prefix = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch + 1)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def accuracy(output, labels, top_k = (1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = labels.size(0)

        indices, pred = output.topk(max_k, 1)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []

        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / (batch_size)))

        return res

# %% Dataset loader
#"""
class ImageNetDataset(Dataset):
    def __init__(self, filenames, dtype = np.float32):
        self.labels = []
        self.parameters = []

        for filename in filenames:
            with open(filename, "rb") as fileptr:
                dataset = pickle.load(fileptr)

                #if "mean" not in dataset:
                #    self.parameters.append(dataset["data"] / 255.0)
                #else:
                #    self.parameters.append((dataset["data"] - dataset["mean"]) / 255.0)

                self.parameters.append(dataset["data"])
                self.labels += [label - 1 for label in dataset["labels"]]

        self.parameters = np.vstack(self.parameters).astype(dtype)
        self.num_classes = max(self.labels) + 1
        self.num_samples = self.parameters.shape[0]
        self.num_features = self.parameters.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #return self.parameters[idx], self.labels[idx]
        return self.parameters[idx]

filenames = glob.glob("%s/Train/*_1" % (data_dir))
train_batch_size = 10
dtype = np.float64
train_dataset = ImageNetDataset(filenames, dtype)
#train_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, True)
train_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, False)

"""
num_epochs = 1
t_total = 0.0

for epoch in range(num_epochs):
    t_start = time.time()

    for idx, (samples, labels) in enumerate(train_loader):
        t_stop = time.time()
        t_total += t_stop - t_start
        t_start = time.time()

print(t_total)
"""

num_epochs = 1
num_batches = (len(train_dataset) + train_batch_size - 1) // train_batch_size

t_start = time.time()

for epoch in range(num_epochs):
    for batch in range(num_batches):
        idx_start = (batch + 0) * train_batch_size
        idx_stop  = min(len(train_dataset), (batch + 1) * train_batch_size)

        #samples, labels = train_dataset[idx_start:idx_stop]
        samples = train_dataset[idx_start:idx_stop]
        test = np.min(samples)
        #print(test)
        #print(samples)
        #break
        #print(labels)
        #break

t_stop = time.time()
t_total = t_stop - t_start
print(t_total)

#filenames = glob.glob("%s/Val/*" % (data_dir))
#val_batch_size = 1024
#val_dataset = ImageNetDataset(filenames, dtype)
#val_loader = torch.utils.data.DataLoader(val_dataset, val_batch_size, False)
#"""

# %%
"""
class IrisDataset(Dataset):
    def __init__(self, filenames, dtype = np.float32):
        self.labels = []
        self.parameters = []

        for filename in filenames:
            dataset = pandas.read_csv(filename)

            self.parameters.append(dataset.values[:, :-1])
            self.labels.append(dataset.values[:, -1])

        self.parameters = np.vstack(self.parameters).astype(dtype)
        self.labels = sklearn.preprocessing.LabelEncoder().fit_transform(np.hstack(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.parameters[idx], self.labels[idx]

train_batch_size = 32
val_batch_size = 1024
split_percentage = 0.2
filenames = glob.glob("%s/*" % (data_dir))
iris_dataset = IrisDataset(filenames)

train_size = int(np.round(len(iris_dataset) * 0.8))
val_size = len(iris_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(iris_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, True)
val_loader = torch.utils.data.DataLoader(val_dataset, val_batch_size, False)
#"""

# %% Model
class Model(nn.Module):
    """
    Simple neural network classification model
    """
    def __init__(self, num_inputs, num_outputs, num_layers, num_nodes_layer, dtype = torch.float32):
        super(Model, self).__init__()

        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.num_nodes_layer = num_nodes_layer

        assert(num_layers >= 1)

        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(num_inputs, num_nodes_layer, dtype = dtype)]
        self.layers += [nn.Linear(num_nodes_layer, num_nodes_layer, dtype = dtype) for _ in range(num_layers - 1)]
        self.layers += [nn.Linear(num_nodes_layer, num_outputs, dtype = dtype)]

    def forward(self, x):
        y = torch.sigmoid(self.layers[0](x))

        for layer in range(1, self.num_layers - 1):
            y = torch.sigmoid(self.layers[layer](y))

        y = nn.functional.softmax(self.layers[-1](y), dim = 1)

        return y

#num_inputs = train_dataset.dataset.parameters.shape[1]
#num_outputs = max(train_dataset.dataset.labels) + 1

num_inputs = train_dataset.parameters.shape[1]
num_outputs = max(train_dataset.labels) + 1

dtype = torch.float64
num_layers = 4
num_nodes_layer = 8
model = Model(num_inputs, num_outputs, num_layers, num_nodes_layer, dtype)
if device.type == "cuda": model.cuda(device)

criterion = nn.CrossEntropyLoss()
if device.type == "cuda": criterion.cuda(device)

learning_rate = 0.1
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum)

# %% Training algorithm
time_breakdown = { "reading": 0.0, "loading": 0.0, "prediction": 0.0, "accuracy": 0.0, "backpropagation": 0.0 }

def train(train_loader, model, criterion, optimizer, epoch = 0, output_freq = 20):
    batch_time = Meter("Time", ":6.3f")
    data_time = Meter("Data", ":8.3g")
    losses = Meter("Loss", ":.4e")
    top_1 = Meter("Acc@1", ":8.4g")
    top_5 = Meter("Acc@5", ":8.4g")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top_1, top_5], prefix = "Epoch: [{}]".format(epoch))

    model.train()

    t_start = time.time()

    for idx, (samples, labels) in enumerate(train_loader):
        t_stop = time.time()
        time_breakdown["reading"] += t_stop - t_start

        # Data loading time
        t_start = time.time()

        if device.type == "cuda":
            samples = samples.cuda(device, non_blocking = False)
            labels  = labels.cuda(device, non_blocking = False)

        t_stop = time.time()
        time_breakdown["loading"] += t_stop - t_start
        data_time.update(time_breakdown["loading"])

        # compute output
        t_start = time.time()

        output = model(samples)
        loss = criterion(output, labels)

        t_stop = time.time()
        time_breakdown["prediction"] += t_stop - t_start

        # Measure accuracy and record loss
        t_start = time.time()

        results = accuracy(output, labels, top_k  = (1, 5))

        t_stop = time.time()
        time_breakdown["accuracy"] += t_stop - t_start

        losses.update(loss.item(), samples.size(0))
        top_1.update(results[0][0], samples.size(0))
        top_5.update(results[1][0], samples.size(0))

        # Compute gradient and do SGD step
        t_start = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_stop = time.time()
        time_breakdown["backpropagation"] += t_stop - t_start

        # Print progress
        if idx % output_freq == 0:
            progress.display(idx)

        # Restart timer
        t_start = time.time()

num_epochs = 4
output_freq = 10

"""
num_epochs = 1
output_freq = 1
#"""

for epoch in range(num_epochs):
    train(train_loader, model, criterion, optimizer, epoch, output_freq)

total_time = sum(time_breakdown.values())

print("      Architecture: %s" % (device.type.upper()))
print("Number of features: %d" % (train_dataset.parameters.shape[1]))
print(" Number of samples: %d" % (train_dataset.parameters.shape[0]))
print("  Number of epochs: %d" % (num_epochs))
print(" Number of weights: %d" % (sum(param.numel() for param in model.parameters() if param.requires_grad)))
print("        Total time: %8.4g s ( %6.2f )" % (total_time, 100.0))
print("           Reading: %8.4g s ( %6.2f )" % (time_breakdown["reading"], time_breakdown["reading"] / total_time * 100.0))
print("           Loading: %8.4g s ( %6.2f )" % (time_breakdown["loading"], time_breakdown["loading"] / total_time * 100.0))
print("        Prediction: %8.4g s ( %6.2f )" % (time_breakdown["prediction"], time_breakdown["prediction"] / total_time * 100.0))
print("   Backpropagation: %8.4g s ( %6.2f )" % (time_breakdown["backpropagation"], time_breakdown["backpropagation"] / total_time * 100.0))

# %% End of program
