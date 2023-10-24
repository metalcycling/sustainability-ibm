#!/usr/bin/env python
# coding: utf-8

# # Tokenizer Training
# 
# We will begin by training a tokenizer on our desired dataset. A small subset of Wikipedia is chosen here for illustrative purposes. HuggingFace provides the tokenizer, and in this case the dataset as well. We will take HuggingFace's base RoBERTa tokenizer and retrain it on WikiText-103.
# 
# *Please note that our sample WikiText-103 dataset is roughly .1% the size of the full Watson-English dataset used to train our provided RoBERTa model. As such, pre-training on this dataset is a toy problem and intended mainly as a functional test and illustrative example - the resulting model will be highly overfitted to third-person, descriptive language with extensive formatting, and should not be expected to perform well as a general-purpose downstream language model. That said, the provided code is set up to scale gracefully to large datasets (including those used to produce our provided RoBERTa model) and we invite users to train models using their own datasets! We recommend at least 5GB of raw data.*

# In[1]:


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


# Large-scale datasets are often split into multiple individual files, or "shards". While WikiText-103 is not truly large enough to merit this treatment, we will shard the dataset manually for illustrative purposes. 
# 
# Shards reside in `./wiki_shards/`

# In[2]:


# Fetch dataset and split into 10 shards for illustration purposes
dataset = load_dataset("wikitext", name = "wikitext-103-raw-v1", split = "train")

if not os.path.exists('wiki_shards/'):
    os.mkdir('wiki_shards')
    
num_shards = 10

for i in range(num_shards):
    shard = dataset[(i + 0) * len(dataset) // num_shards : (i + 1) * len(dataset) // num_shards]["text"]
    
    with open("wiki_shards/wiki_shard_%d.pkl" % (i), "wb") as fileptr:
        pickle.dump(shard, fileptr)


# In[3]:


# Define an iterator over our dataset that returns batches of 64 lines at a time
shards = os.listdir("./wiki_shards")
min_sentence_length = 10

def batch_iterator():
    for shard in shards:
        with open("./wiki_shards/%s" % (shard), "rb") as fileptr:
            data = pickle.load(fileptr)
            
        batch = []
        
        for line in data:
            if len(line) > min_sentence_length: # remove trivially short sentences
                batch.append(line)
                
            if len(batch) == 64: # If batch is of size 64 return it
                yield batch
                batch = []
                
        yield batch
        
        print("Shard '%s' completed" % (shard))


# In[11]:


dataset["text"][3]
print(dataset["text"][3])
len(dataset["text"][3].split())


# In[4]:


shard = "wiki_shard_0.pkl"
with open("./wiki_shards/%s" % (shard), "rb") as fileptr:
    data = pickle.load(fileptr)


# In[13]:


data[10]


# In[4]:


# Create and train our tokenizer
vocab = 50000 + 256 + 5 # 50K learned tokens, 256 base characters, 5 dummy tokens

# Get a BPE tokenizer with the right preprocessors, which we'll then retrain
tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size = vocab)
tokenizer.save_pretrained("roberta-tokenizer")

print("Training complete!")


# Now that training is done, let's take a look at how the tokenizer handles a real sentence, and how it splits apart words it does not recognize:

# In[5]:


# Test run for our new tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-tokenizer")

sample = "This was a triumph. I'm making a note here: HUGE SUCCESS! It's hard to overstate my satisfaction."
print(sample)

tokens = tokenizer(sample)["input_ids"]
print(tokens)

print("\n".join([tokenizer.decode(token) for token in tokens]))


# # Pre-Tokenizing the Dataset
# 
# Now that we have a trained tokenizer, let's convert our dataset into token indices. At the same time, we'll pack sequences together until we exceed the maximum RoBERTa sequence length of 512. At that point we back off, pad out the sequence, and write that batch to our output shards. By tokenizing, packing, and padding sequences all in advance, we can compress our dataset size, accelerate downstream dataloading, and streamline our training procedure.
# 
# This time we will use Ray for parallelism - two workers will iterate over their respective input shards, and each will write a single output shard that we'll use for training. Currently we only support scaling on single nodes - do not attempt to run this notebook on OpenShift!
# 
# *(Parallelism on OpenShift for both tokenizer training and preprocessing will be provided in a future release)*
# 
# Output shards reside in `./wiki_processed_shards/`

# In[19]:


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
    buffer = [] # our write buffer
    counter = 0 # written batch counter
    ntrunc = 0
    npad = []
    
    with pa.ipc.new_file("%s/wiki_processed_shards/shard_%03d.arrow" % (directory, session.get_world_rank()), schema) as writer:
        for j in range(len(subsets)): # for each input shard
            filename = subsets[j]
            
            with open("%s/wiki_shards/%s" % (directory, filename), "rb") as fileptr:
                dataset = pickle.load(fileptr)
                
            for entry in dataset:
                line = tokenizer(entry)["input_ids"] # tokenize!
                
                if len(line) > 5: # Ignore short sentences
                    if len(line) > seq_len: # Truncate long sentences
                        line = line[:seq_len - 1] + [eos]
                        ntrunc += 1
        
                    if len(buffer) + len(line) <= seq_len: # If line fits into buffer, add it
                        buffer += line
                    else: 
                        # Else, pad out buffer
                        npad.append(seq_len - len(buffer))
                        buffer += [pad,] * (seq_len - len(buffer))
                        
                        # Write buffer. We subtract 25K to prevent overflow - 
                        # int16 only goes up to 32767, vocab size is >50K
                        batch = pa.record_batch([pa.array([x - 25000 for x in buffer], pa.int16())], schema)
                        writer.write(batch)
                        counter += 1
                        
                        # Clear buffer and write line
                        buffer = line
                        
            print("Shard %d: %d of %d complete, length = %d, avg pad = %f" % (session.get_world_rank(), j + 1, len(subsets), counter, sum(npad) / len(npad)))
                        
            session.report({"training_iteration": j + 1})
            
        # Write final buffer
        buffer += [pad,] * (seq_len - len(buffer))
        npad.append(seq_len - len(buffer))
        batch = pa.record_batch([pa.array([x - 25000 for x in buffer], pa.int16())], schema)
        writer.write(batch)
        counter += 1
        
    writer.close()
    
    print("Shard %d complete, final length = %d lines, with %f pads per %d sequence tokens and %d truncations" % (session.get_world_rank(), counter, sum(npad) / len(npad), seq_len, ntrunc))


# In[20]:


# Run our Ray-based pre-tokenizer!
if not os.path.exists("./wiki_processed_shards"):
    os.mkdir("wiki_processed_shards")
    
# For illustrative purposes, we'll condense our 10 input shards into 2 output shards
trainer = TorchTrainer(train_loop_per_worker = process, train_loop_config = { "shards": os.listdir("./wiki_shards"), "directory": os.getcwd() }, scaling_config = ScalingConfig(num_workers = 2))
trainer.fit()

print('Preprocessing complete!')


# In[21]:


# Original dataset size:
orig_size = sum([os.path.getsize("./wiki_shards/%s" % (filename)) for filename in os.listdir("./wiki_shards")])
print("Original dataset size (pickle-compressed): %s MB" % (orig_size >> 20))


# In[22]:


# New dataset size:
new_size = sum([os.path.getsize("./wiki_processed_shards/%s" % (filename)) for filename in os.listdir("./wiki_processed_shards")])
print("Tokenized dataset size: %s MB" % (new_size >> 20))


# *For large datasets, we have observed a roughly 3-4x reduction in tokenized dataset size compared to raw text. 
# For comparison, we've found this to be slightly better than pickle, and slightly worse than gzip.*
