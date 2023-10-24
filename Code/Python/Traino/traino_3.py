# %% Image classification
import sys

sys.path.append("/home/metalcycling/Documents/CodeFlare_Shared_Memory/Code/Master/build/sustainability")

from codeflare_shared_memory import *
import time
import glob
import pickle
import pandas
import sklearn
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
from utilities import *

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
from torch.utils.data import Dataset

# %% Runtime parameters
#device = torch.device("cuda:0")
device = torch.device("cpu")
data_dir = "/home/metalcycling/Documents/Sustainability/Data/ImageNet/8x8"
#data_dir = "/home/metalcycling/Documents/Sustainability/Data/ImageNet/16x16"
#data_dir = "/home/metalcycling/Documents/Sustainability/Data/Iris"

# %% Dataset loader
#"""
class ImageNetDataset(Dataset):
    def __init__(self, filenames, dtype = np.float32):
        self.segment_size = 1024 ** 3
        self.segment_name = "traino"
        self.shared_memory = CodeFlareSharedMemory(self.segment_name, self.segment_size)

        if not self.shared_memory.is_found():
            self.num_samples = 0

            for filename in filenames:
                with open(filename, "rb") as fileptr:
                    dataset = pickle.load(fileptr)
                    self.num_features = dataset["data"].shape[1]
                    self.num_samples += dataset["data"].shape[0]

            self.shared_memory.add_table("parameters", self.num_samples, self.num_features)
            self.shared_memory.add_table("labels", self.num_samples, 1)

            for filename in filenames:
                with open(filename, "rb") as fileptr:
                    dataset = pickle.load(fileptr)

                    for idx, row in enumerate(dataset["data"]):
                        self.shared_memory.add_row("parameters", list(row))
                        self.shared_memory.add_row("labels", [dataset["labels"][idx] - 1])

        else:
            self.num_samples = self.shared_memory.get_num_rows("parameters")
            self.num_features = self.shared_memory.get_num_cols("parameters")

        #self.num_classes = int(np.array(self.shared_memory.get_rows("labels", 0, self.num_samples), copy = False).max()) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        if isinstance(idx, slice):
            #start = idx.start if idx.start != None else 0
            #stop  = idx.stop if idx.stop != None else self.num_samples
            start = idx.start
            stop = idx.stop
            #labels = np.array(self.shared_memory.get_rows("labels", start, stop), copy = False).reshape(-1).astype(int)
            #labels = np.array(self.shared_memory.get_rows("labels", start, stop), copy = False)
            parameters = np.array(self.shared_memory.get_rows("parameters", start, stop), copy = False)

        else:
            start = idx
            stop  = start + 1
            labels = int(np.array(self.shared_memory.get_rows("labels", start, stop), copy = False)[0])
            parameters = np.array(self.shared_memory.get_rows("parameters", start, stop), copy = False)[0]

        #return parameters, labels
        return parameters
        """
        return np.array(self.shared_memory.get_rows("parameters", idx.start, idx.stop), copy = False)
        #return self.shared_memory.get_rows("parameters", idx.start, idx.stop)
        #return self.shared_memory.get_num_rows("parameters")
        #return None

    def remove(self):
        self.shared_memory.remove()

filenames = glob.glob("%s/Train/*_1" % (data_dir))
train_batch_size = 10
train_dataset = ImageNetDataset(filenames)
#train_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, True)
#train_loader = torch.utils.data.DataLoader(train_dataset, train_batch_size, False)

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
        idx_stop = min(len(train_dataset), (batch + 1) * train_batch_size)

        #samples, labels = train_dataset[idx_start:idx_stop]
        samples = train_dataset[idx_start:idx_stop]
        test = np.min(samples)
        #print(test)
        #break
        #samples = train_dataset[idx_start, idx_stop]
        #print(labels)
        #break

        #num_samples = train_dataset.shared_memory.get_num_rows("parameters")

t_stop = time.time()
t_total = t_stop - t_start
print(t_total)

#filenames = glob.glob("%s/Val/*" % (data_dir))
#val_batch_size = 1024
#val_dataset = ImageNetDataset(filenames)
#val_loader = torch.utils.data.DataLoader(val_dataset, val_batch_size, False)
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

num_inputs = train_dataset.num_features
num_outputs = train_dataset.num_classes

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

num_epochs = 1
output_freq = 10

"""
num_epochs = 1
output_freq = 1
#"""

for epoch in range(num_epochs):
    train(train_loader, model, criterion, optimizer, epoch, output_freq)

#"""
total_time = sum(time_breakdown.values())

print("      Architecture: %s" % (device.type.upper()))
print("Number of features: %d" % (train_dataset.num_features))
print(" Number of samples: %d" % (train_dataset.num_samples))
print("  Number of epochs: %d" % (num_epochs))
print(" Number of weights: %d" % (sum(param.numel() for param in model.parameters() if param.requires_grad)))
print("        Total time: %8.4g s ( %6.2f )" % (total_time, 100.0))
print("           Reading: %8.4g s ( %6.2f )" % (time_breakdown["reading"], time_breakdown["reading"] / total_time * 100.0))
print("           Loading: %8.4g s ( %6.2f )" % (time_breakdown["loading"], time_breakdown["loading"] / total_time * 100.0))
print("        Prediction: %8.4g s ( %6.2f )" % (time_breakdown["prediction"], time_breakdown["prediction"] / total_time * 100.0))
print("   Backpropagation: %8.4g s ( %6.2f )" % (time_breakdown["backpropagation"], time_breakdown["backpropagation"] / total_time * 100.0))
#"""

# %% End of program
