# %% Image classification
import sys

sys.path.append("/home/metalcycling/Documents/CodeFlare_Shared_Memory/Code/Python/Reading/v2")

from codeflare_shared_memory import *
import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
num_rows = 100000

fileptr = open("results.dat", "w")

for num_cols in np.logspace(0, 10, 11, base = 2, dtype = int):
    table_name = "samples"
    segment_size = 1024 ** 3
    segment_name = "traino"
    shared_memory = CodeFlareSharedMemory(segment_name, segment_size)
    shared_memory.add_table(table_name, num_rows, num_cols)

    if shared_memory.is_found:
        num_rows = shared_memory.get_num_rows(table_name)
        num_cols = shared_memory.get_num_cols(table_name)

        samples = np.empty((num_rows, num_cols))
        samples[:] = np.array(shared_memory.get_rows(table_name, 0, num_rows), copy = False)

    else:
        samples = np.random.uniform(size = (num_rows, num_cols))

        for sample in samples:
            shared_memory.add_row(table_name, list(sample))

    num_epochs = 8

    for batch_size in np.logspace(0, 16, 17, base = 2, dtype = int):
        num_batches = (num_rows + batch_size - 1) // batch_size
        mean = 0.0

        # Numpy
        t_start = time.time()

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                start = batch * batch_size
                stop = min(num_rows, (batch + 1) * batch_size)
                sample = samples[start:stop]
                mean += sample.mean()

        t_stop = time.time()
        t_numpy = t_stop - t_start

        # CFSHM
        t_start = time.time()

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                start = batch * batch_size
                stop = min(num_rows, (batch + 1) * batch_size)
                sample = np.array(shared_memory.get_rows(table_name, start, stop), copy = False)
                mean += sample.mean()

        t_stop = time.time()
        t_cfshm = t_stop - t_start

        fileptr.write("%d %d %f %f\n" % (num_cols, batch_size, t_numpy, t_cfshm))

    shared_memory.remove()

fileptr.close()

# %% End of program
