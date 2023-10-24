"""
RoBERTa training with PyTorch infrastructur

%load_ext autoreload
%autoreload 2
"""

# %% Modules

import os
import math
import time
import argparse
import functools
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from fm_nlp.architecture import RoBERTa
from fm_nlp.pretraining.mlm import mask_mlm
from torch.nn.parallel import DistributedDataParallel as DDP

from fm import data as fmdata
from fm import utils
from fm.modules import Alibi
from fm.utils import (
    ExperimentsTracker,
    get_local_rank,
    get_world_size,
    run_rank_n,
    setup_distributed,
)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# %% Command line arguments

parser = argparse.ArgumentParser(description = "RoBERTa on IBM NLP data for OCP")

parser.add_argument("--simulated_gpus", type = int, default = 128, help = "Number of GPUs to simulate")
parser.add_argument("--b_size", type = int, default = 96, help = "Batch size per GPU")
parser.add_argument("--num_steps", type = int, default = 200000, help = "Number of SGD steps.")
parser.add_argument("--datapath", type = str, default = "/data/input/roberta_v3/", help = "Dataset directory (absolute path)")
parser.add_argument("--experiment_name", type = str, required = True, help = "Experiment name")
parser.add_argument("--aim_repo", default = "/data/output/aim_repo", type = str, help = "Aim repository")
parser.add_argument("--logdir", type = str, default = "/data/output/roberta-torchnative-alibi/", help = "Log/checkpoint directory (absolute path)",)
parser.add_argument("--num_ckps", type = int, default = 3, help = "Number of checkpoints to maintain (n most recent)")
parser.add_argument("--ckp_interval", type = int, default = 5000, help = "Checkpoint interval")
parser.add_argument("--report_interval", type = int, default = 100, help = "Reporting interval")
parser.add_argument("--vocab", type = int, default = 50260, help = "Size of tokenizer")
parser.add_argument("--max_pos", type = int, default = 512, help = "Max input sequence length")
parser.add_argument("--aggressive_cache", action = "store_true", help = "Pre-fill data cache?")
parser.add_argument("--reset_stepcount", action = "store_true", help = "If loading from checkpoint, force restart to step 0?")

args = parser.parse_args()

# %% TEMP

args = []
args += ["--experiment_name", "roberta-test"]
args += ["--simulated_gpus", "1"]

"""
args += ["--b_size", "8"]
args += ["--num_steps", "4000"]
args += ["--reset_stepcount"]
#"""

"""
args += ["--b_size", "16"]
args += ["--num_steps", "4000"]
#"""

"""
args += ["--b_size", "32"]
args += ["--num_steps", "4000"]
#"""

#"""
args += ["--b_size", "64"]
args += ["--num_steps", "4000"]
#"""

"""
args += ["--b_size", "8"]
args += ["--num_steps", "4000"]
args += ["--reset_stepcount"]
#"""

"""
args += ["--b_size", "16"]
args += ["--num_steps", "4000"]
args += ["--reset_stepcount"]
#"""

"""
args += ["--b_size", "32"]
args += ["--num_steps", "4000"]
args += ["--reset_stepcount"]
#"""

"""
args += ["--b_size", "64"]
args += ["--num_steps", "4000"]
args += ["--reset_stepcount"]
#"""

args += ["--vocab", "50261"]
args += ["--datapath", "/workspace/data/pedro/input/wiki"]
args += ["--aim_repo", "/workspace/data/pedro/output/wiki"]
args += ["--logdir", "/workspace/data/pedro/output/wiki"]
args = parser.parse_args(args = args)

# %%

# For now, changing other hyperparameters will require changing the hardcoded constants in the code below

attn_mask_lvl = (
    0  # Use 0 for no masking, 1 to mask off pad tokens, or 2 to mask off subsequences from each other in each line
)
bos_token = 0
pad_token = 1
eos_token = 2
mask_token = 3
n_start_dummies = 4
n_end_dummies = 0
token_offset = 25000
vocab_size = args.vocab
max_pos = args.max_pos

aggressive_caching = args.aggressive_cache
is_caching = True
aggressive_caching = is_caching and aggressive_caching
assert (
    args.simulated_gpus % get_world_size() == 0
), f"Simulated gpus {args.simulated_gpus} must be an exact multiple of actual gpus {get_world_size()}"
assert (
    args.simulated_gpus >= get_world_size()
), "Cannot simulate fewer gpus than are available. Please lower gpu count to match the desired target value."
assert (
    args.ckp_interval % args.report_interval == 0
), f"For accurate timing metrics, reporting interval {args.report_interval} must divide checkpoint interval {args.ckp_interval} evenly"

mask_fn = lambda x: None
if attn_mask_lvl == 1:
    mask_fn = functools.partial(utils.pad_mask, pad=pad_token)
elif attn_mask_lvl == 2:
    mask_fn = functools.partial(utils.sequence_mask, pad=pad_token, eos=eos_token)
elif attn_mask_lvl != 0:
    print(f"Warning: specified attention masking level {attn_mask_lvl} does not exist. Defaulting to 0 (no masking)")

torch.cuda.set_device(get_local_rank())
setup_distributed()

run_rank_n(os.makedirs, barrier=True)(args.logdir, exist_ok=True)

logger = ExperimentsTracker(__name__, args.experiment_name, args.aim_repo, additional_logdir=args.logdir)
logger.log_args(args)

logger.info(
    "Starting training run:",
    gpus=get_world_size(),
    simulated_gpus=args.simulated_gpus,
    num_steps=args.num_steps,
    data_path=args.datapath,
    log_dir=args.logdir,
    pre_caching=aggressive_caching,
)


# Training loop
def train_func(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.log_hyperparam("allow_tf32", True)

    emu_factor = args.simulated_gpus // get_world_size()
    logger.log_hyperparam("emu_factor", emu_factor)

    bsize = args.b_size
    effective_bsize = args.b_size * args.simulated_gpus
    logger.log_hyperparam("effective_bsize", effective_bsize)
    start_step = 0

    # Model
    model = RoBERTa(
        vocab_size, 768, 64, 64, mask_on_none=False, use_mean=False, pos_embed_fn=Alibi
    )  # Change up your model architecture here!
    #     model = RoBERTa(vocab_size, 512, 64, 64, mask_on_none=False,
    #                     use_mean=False, nheads=16, nlayers=24, pos_embed_fn=Alibi) # Change up your model architecture here!
    logger.info("Models spawned, attempting DDP synchronization...")
    num_params = utils.pcount(model)
    model = DDP(
        model.to(get_local_rank()),
        device_ids=[get_local_rank()],
        output_device=get_local_rank(),
        gradient_as_bucket_view=True,
    )
    loss_fn = nn.CrossEntropyLoss()
    logger.info("Model created!", num_params=num_params)

    # Optimizers
    lr = 1.5e-5 * (2 * math.sqrt(effective_bsize / 20) - 1)
    logger.info(lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: min(x / (args.num_steps / 20), (args.num_steps - x) / args.num_steps)
    )

    # Load checkpoint
    checkpoint = utils.get_latest(os.path.join(args.logdir, "checkpoints/"))
    if checkpoint:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        if not args.reset_stepcount:
            start_step = checkpoint_data.get("step")
            optimizer.load_state_dict(checkpoint_data.get("optimizer_state"))
            scheduler.load_state_dict(checkpoint_data.get("scheduler_state"))
        logger.info("Prior checkpoint", checkpoint, "detected. Loading from step:", start_step)
        model.module.load_state_dict(checkpoint_data.get("model_state"))
        model.module.to(get_local_rank())
    else:
        logger.info("No checkpoint detected, starting from scratch")

    # Prepare to save checkpoints
    checkpointer = utils.Checkpointer(args.logdir, args.num_ckps)

    # Data
    convert_fn = lambda x: utils.pyarrow_idx_to_torch(x, offset=token_offset)
    aug_fn = lambda x: mask_mlm(
        x, token=mask_token, vsize=vocab_size, n_dummies_start=n_start_dummies, n_dummies_end=n_end_dummies
    )

    def make_dataset(is_val):
        data = fmdata.Shard_Dataset(args.datapath, trainsplit=0.999, is_val=is_val)
        if is_caching:
            data = fmdata.Cache_Dataset(data)
        data = fmdata.Preprocess_Dataset(data, convert_fn)
        data = fmdata.Preprocess_Dataset(data, aug_fn)
        return data

    train_data = make_dataset(is_val=False)
    val_data = make_dataset(is_val=True)

    train_loader = fmdata.Distributed_Dataloader(train_data, bsize=bsize, step=start_step)
    val_loader = fmdata.Distributed_Dataloader(val_data, bsize=bsize)
    logger.info("Datasets constructed!", effective_bsize=effective_bsize)

    # Cache data
    if aggressive_caching:
        logger.info("Beginning caching. For large datasets, this may take ~1 min per 10 GB.")
        train_loader.dataset.set_fn(False)  # Disable augmentation for dummy pass
        val_loader.dataset.set_fn(False)

        start = time.time()
        for inp in val_loader:
            pass
        logger.info(msg="Valset cached!", cache_time_in_minutes=(time.time() - start) / 60, length=len(val_loader))

        start = time.time()
        report_steps = {(len(train_loader) * (j + 1)) // 5: j + 1 for j in range(5)}
        for i, inp in enumerate(train_loader):
            if (i + 1) in report_steps:
                logger.info("Trainset", 20 * report_steps[i + 1], "percent complete")
        logger.info(msg="Trainset cached!", cache_time_in_minutes=(time.time() - start) / 60, length=len(train_loader))

        train_loader.dataset.set_fn(True)  # Re-enable augmentation for future passes
        val_loader.dataset.set_fn(True)
        train_loader.dataset.set_flags(data=None)  # Wipe pyarrow readers - probably unnecessary but good bookkeeping
        val_loader.dataset.set_flags(data=None)

    # Train
    logger.info(
        "Beginning training! If using a large dataset w/o aggressive caching, may take ~1 min per 20GB before starting.",
        num_steps=args.num_steps - start_step,
        effective_epochs=(args.num_steps - start_step) * emu_factor / len(train_loader),
    )
    train_loader = utils.cycle(train_loader)
    model.train()
    losstracker = torch.zeros(1).to(get_local_rank())
    trackertracker = 0
    start = time.time()
    for step in range(start_step, args.num_steps):
        #"""
        if step == 4000 - 0:
            break
        #"""
        optimizer.zero_grad()
        if step == start_step:
            dist.barrier()
            logger.info("Workers synchronized!")
        for ministep in range(emu_factor):
            with model.no_sync() if (ministep + 1) != emu_factor else nullcontext():
                inp, labels = next(train_loader)
                inp = inp.to(get_local_rank())
                labels = labels.to(get_local_rank())
                inds = torch.nonzero(labels.view(-1) + 100).squeeze(1)  # Pick out targets. -100 is the [IGNORE] index
                mask = mask_fn(inp)
                pred = model(inp, inds, enc_mask=mask)
                loss = loss_fn(pred, labels.view(-1)[inds]).div(emu_factor)
                losstracker += loss.item()
                if step == start_step and ministep == 0:
                    dist.barrier()
                    logger.info("Got through initial forward pass")
                loss.backward()
                if step == start_step and ministep == 0:
                    dist.barrier()
                    logger.info("Got through initial backward pass")
        nn.utils.clip_grad_norm_(model.parameters(), 1 / math.sqrt(effective_bsize))
        optimizer.step()
        scheduler.step()
        trackertracker += 1

        # Report training loss and speed
        if (step + 1) % args.report_interval == 0:
            dist.all_reduce(losstracker, op=dist.ReduceOp.SUM)
            logger.info(
                step=step + 1,
                trainloss=losstracker.item() / trackertracker / get_world_size(),
                speed=(time.time() - start) / trackertracker,
            )
            logger.track(
                losstracker.item() / trackertracker / get_world_size(), "loss", step + 1, context={"context": "train"}
            )
            logger.track(scheduler.get_lr()[0], "learning_rate", step + 1)
            losstracker.zero_()
            trackertracker = 0
            start = time.time()

        # Validate and checkpoint model
        if (step + 1) % args.ckp_interval == 0:
            model.eval()
            losstracker = torch.zeros(1).to(get_local_rank())
            for inp, labels in val_loader:
                inp = inp.to(get_local_rank())
                labels = labels.to(get_local_rank())
                with torch.no_grad():
                    inds = torch.nonzero(labels.view(-1) + 100).squeeze(1)
                    mask = mask_fn(inp)
                    pred = model(inp, inds, enc_mask=mask)
                    loss = loss_fn(pred, labels.view(-1)[inds])
                losstracker += loss.item()
            dist.all_reduce(losstracker, op=dist.ReduceOp.SUM)
            valloss = losstracker.item() / len(val_loader) / get_world_size()
            logger.info("Checkpointing!", step=step + 1, valloss=valloss)
            logger.track(
                losstracker.item() / len(val_loader) / get_world_size(), "loss", step + 1, context={"subset": "val"}
            )
            overwritten = checkpointer.save(
                step=step + 1,
                model_state=model.module.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                loss=valloss,
            )
            if overwritten:
                logger.info("Checkpoint", overwritten, "overwritten")
            model.train()
            start = time.time()
            losstracker.zero_()

    logger.info(msg="Writing final checkpoint", step=step + 1)
    if args.num_ckps > 0:
        overwritten = checkpointer.save(
            step=step + 1,
            model_state=model.module.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            loss=None,
            final=True,
        )
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

start = time.time()
train_func(args)
logger.info("Job Complete!", total_hours=(time.time() - start) / 3600)

# %% End of program
