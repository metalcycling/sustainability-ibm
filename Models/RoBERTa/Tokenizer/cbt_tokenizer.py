"""
CBT Tokenizer Training:

We will begin by training a tokenizer on our desired dataset. The Children's Book Test (CBT) dataset is used in this example. HuggingFace provides the tokenizer, and in this case the dataset as well. We will take HuggingFace's base RoBERTa tokenizer and retrain it on CBT.
"""

# %% Modules

import os
import time
import pickle

import pyarrow as pa
from pyarrow import fs

from datasets import load_dataset
from transformers import AutoTokenizer

import ray
from ray.air import session
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer

# %% Step 1:

#
# Large-scale datasets are often split into multiple individual files, or "shards". While our dataset is not truly large enough to merit this treatment, we will shard the dataset manually for illustrative purposes.
#

# Fetch dataset and split into 10 shards for illustration purposes
dataset = load_dataset("cbt", name = "CN", split = "train")

if not os.path.exists("cbt_shards/"):
    os.mkdir("cbt_shards")

num_shards = 10

for i in range(num_shards):
    shard = dataset[(i + 0) * len(dataset) // num_shards : (i + 1) * len(dataset) // num_shards]["sentences"]

    with open("cbt_shards/cbt_shard_%d.pkl" % (i), "wb") as fileptr:
        pickle.dump(shard, fileptr)

# %% Step 2

#
# Define an iterator over our dataset that returns batches of 64 lines at a time
#

shards = os.listdir("./cbt_shards")
min_paragraph_length = 10

def batch_iterator():
    for shard in shards:
        with open("./cbt_shards/%s" % (shard), "rb") as fileptr:
            dataset = pickle.load(fileptr)

        batch = []

        for idx in range(len(dataset)):
            # This dataset has entries that are repeated for downtuning operations. We don't need that repetition for the FM training.
            if idx > 0:
                if dataset[idx][0] == first_word and dataset[idx][1] == second_word and dataset[idx][2] == third_word:
                    continue
                else:
                    first_word = dataset[idx][0]
                    second_word = dataset[idx][1]
                    third_word = dataset[idx][2]
            else:
                first_word = dataset[idx][0]
                second_word = dataset[idx][1]
                third_word = dataset[idx][2]

            paragraph = " ".join(dataset[idx])

            if len(paragraph) > min_paragraph_length: # remove trivially short paragraphs
                batch.append(paragraph)

            if len(batch) == 64: # If batch is of size 64 return it
                yield batch
                batch = []

        yield batch

        print("Shard '%s' completed" % (shard))

# %% Step 3

#
# Create and train our tokenizer
#

vocab = 50000 + 256 + 5 # 50K learned tokens, 256 base characters, 5 dummy tokens

# Get a BPE tokenizer with the right preprocessors, which we'll then retrain
tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size = vocab)
tokenizer.save_pretrained("roberta-tokenizer")

print("Training complete!")

# %% Step 4

#
# Now that training is done, let's take a look at how the tokenizer handles a real sentence, and how it splits apart words it does not recognize:
#


# Test run for our new tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-tokenizer")

sample = "This was a triumph. I'm making a note here: HUGE SUCCESS! It's hard to overstate my satisfaction."
print(sample)

tokens = tokenizer(sample)["input_ids"]
print(tokens)

print("\n".join([tokenizer.decode(token) for token in tokens]))


# %% Step 5

#
# Pre-Tokenizing the Dataset
# 
# Now that we have a trained tokenizer, let's convert our dataset into token indices. At the same time, we'll pack sequences together until we exceed the maximum RoBERTa sequence length of 512. At that point we back off, pad out the sequence, and write that batch to our output shards. By tokenizing, packing, and padding sequences all in advance, we can compress our dataset size, accelerate downstream dataloading, and streamline our training procedure.
# 
# This time we will use Ray for parallelism - two workers will iterate over their respective input shards, and each will write a single output shard that we'll use for training. Currently we only support scaling on single nodes - do not attempt to run this notebook on OpenShift!
# 

# Define the workload that each parallel Ray actor will run
def process(config):
    shards = config["shards"]
    directory = config["directory"]

    seq_len = 512
    shard = session.get_world_rank()

    # Distribute input shards over workers, each of which produces a single output shard
    subsets = shards[(shard + 0) * len(shards) // session.get_world_size() : (shard + 1) * len(shards) // session.get_world_size()] 

    # LOAD OUR PRETRAINED TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained("%s/roberta-tokenizer" % (directory))
    schema = pa.schema([pa.field("nums", pa.int16())])
    pad = tokenizer("<pad>")["input_ids"][1]
    eos = tokenizer("</s>")["input_ids"][1]

    # ITERATE OVER INPUT FILES, WRITE BATCHES 512 TOKENS AT A TIME TO OUR DATASET SHARD
    output_buffer = [] # our write buffer
    counter = 0 # written batch counter
    ntrunc = 0
    npad = []

    with pa.ipc.new_file("%s/cbt_processed_shards/shard_%03d.arrow" % (directory, session.get_world_rank()), schema) as writer:
        for j in range(len(subsets)): # for each input shard
            filename = subsets[j]

            with open("%s/cbt_shards/%s" % (directory, filename), "rb") as fileptr:
                dataset = pickle.load(fileptr)

            for idx in range(len(dataset)):
                # This dataset has entries that are repeated for downtuning operations. We don't need that repetition for the FM training.
                if idx > 0:
                    if dataset[idx][0] == first_word and dataset[idx][1] == second_word and dataset[idx][2] == third_word:
                        continue
                    else:
                        first_word = dataset[idx][0]
                        second_word = dataset[idx][1]
                        third_word = dataset[idx][2]
                else:
                    first_word = dataset[idx][0]
                    second_word = dataset[idx][1]
                    third_word = dataset[idx][2]

                paragraph = " ".join(dataset[idx])
                tokens = tokenizer(paragraph)["input_ids"] # tokenize!

                if len(tokens) > 5: # Ignore short sentences
                    if len(tokens) > seq_len: # Truncate long sentences
                        tokens = tokens[:seq_len - 1] + [eos]
                        ntrunc += 1

                    if len(output_buffer) + len(tokens) <= seq_len: # If line fits into output_buffer, add it
                        output_buffer += tokens
                    else: 
                        # Else, pad out output_buffer
                        npad.append(seq_len - len(output_buffer))
                        output_buffer += [pad,] * (seq_len - len(output_buffer))

                        # Write output_buffer. We subtract 25K to prevent overflow - 
                        # int16 only goes up to 32767, vocab size is >50K
                        batch = pa.record_batch([pa.array([x - 25000 for x in output_buffer], pa.int16())], schema)
                        writer.write(batch)
                        counter += 1

                        # Clear output_buffer and write tokens
                        output_buffer = tokens

            print("Shard %d: %d of %d complete, length = %d, avg pad = %f" % (session.get_world_rank(), j + 1, len(subsets), counter, sum(npad) / len(npad)))

            session.report({"training_iteration": j + 1})

        # Write final output_buffer
        output_buffer += [pad,] * (seq_len - len(output_buffer))
        npad.append(seq_len - len(output_buffer))
        batch = pa.record_batch([pa.array([x - 25000 for x in output_buffer], pa.int16())], schema)
        writer.write(batch)
        counter += 1

    writer.close()

    print("Shard %d complete, final length = %d lines, with %f pads per %d sequence tokens and %d truncations" % (session.get_world_rank(), counter, sum(npad) / len(npad), seq_len, ntrunc))

# Run our Ray-based pre-tokenizer!
if not os.path.exists("./cbt_processed_shards"):
    os.mkdir("cbt_processed_shards")
    
# For illustrative purposes, we'll condense our 10 input shards into 2 output shards
num_workers = 2
trainer = TorchTrainer(train_loop_per_worker = process, train_loop_config = { "shards": os.listdir("./cbt_shards"), "directory": os.getcwd() }, scaling_config = ScalingConfig(num_workers = num_workers))
trainer.fit()

print('Preprocessing complete!')

# %% Step 6

#
# Check the memory size of the resulting dataset options
#
# *For large datasets, we have observed a roughly 3-4x reduction in tokenized dataset size compared to raw text. 
# For comparison, we've found this to be slightly better than pickle, and slightly worse than gzip.*
#

# Original dataset size:
orig_size = sum([os.path.getsize("./cbt_shards/%s" % (filename)) for filename in os.listdir("./cbt_shards")])
print("Original dataset size (pickle-compressed): %s MB" % (orig_size >> 20))

# New dataset size:
new_size = sum([os.path.getsize("./cbt_processed_shards/%s" % (filename)) for filename in os.listdir("./cbt_processed_shards")])
print("Tokenized dataset size: %s MB" % (new_size >> 20))

# %% End of program
