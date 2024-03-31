"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import pickle
from contextlib import nullcontext
from typing import Literal

import numpy as np
import torch
import torch.amp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim import AdamW
from torch.optim.optimizer import StateDict
from config import DDPConfig, NanoGPTConfig

from model import GPTConfig, GPT
from dataclasses import dataclass

# -----------------------------------------------------------------------------
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
def get_ddp_config(config: NanoGPTConfig) -> DDPConfig | None:
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=config.project.ddp_backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert config.training.gradient_accumulation_steps % ddp_world_size == 0
        config.training.gradient_accumulation_steps //= ddp_world_size
        return DDPConfig(enabled=True, rank=ddp_rank, local_rank=ddp_local_rank, world_size=ddp_world_size)
    else:
        # if not ddp, we are running on a single gpu, and one process
        return None



def get_batch(split: Literal["train", "val"], config: NanoGPTConfig):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    batch_size = config.training.batch_size
    block_size = config.model.context_size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if config.device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)

# attempt to derive vocab_size from the dataset
def get_vocab_size(config: NanoGPTConfig) -> int | None:
    meta_path = os.path.join(config.data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    return meta_vocab_size

@dataclass
class ModelTrainer:
    model: GPT | DDP
    model_args: dict
    scaler: GradScaler
    optimizer: AdamW
    config: NanoGPTConfig
    current_learning_rate: float
    iter_num: int = 0
    best_val_loss: float = 1e9
    running_mfu: float = -1.0
    unoptimized_model: GPT | None = None

    @property
    def raw_model(self):
        return self.model.module if self.config.ddp else self.model # unwrap DDP container if needed

# model init
def _create_model(init_from: str, config: NanoGPTConfig) -> tuple[GPT, int, float, StateDict | None, dict]:
    iter_num: int = 0
    best_val_loss: float = 1e9
    checkpoint = None
    meta_vocab_size = get_vocab_size(config)
    model_args = dict(
        n_layer=config.model.layers,
        n_head=config.model.heads,
        n_embd=config.model.embedding_dimension,
        block_size=config.model.context_size,
        bias=config.model.bias,
        vocab_size=None,
        dropout=config.training.dropout
    ) # start with model_args from command line

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {config.output_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config.output_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        raise Exception("Must be 'scratch' or 'resume'")

    # crop down the model block size if desired, using model surgery
    if config.model.context_size < model.config.block_size:
        model.crop_block_size(config.model.context_size)
        model_args['block_size'] = config.model.context_size # so that the checkpoint will have the right value
    model.to(config.device)
    return (
        model,
        iter_num,
        best_val_loss,
        checkpoint["optimizer"] if checkpoint is not None else None,
        model_args,
    )


def init_model(init_from: str, config: NanoGPTConfig) -> ModelTrainer:
    model, iter_num, best_val_loss, optimizer_settings, model_args = _create_model(init_from, config)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(
        config.training.adamw.weight_decay,
        config.training.adamw.learning_rate,
        (config.training.adamw.beta1, config.training.adamw.beta2),
        config.device_type
    )
    if optimizer_settings is not None:
        optimizer.load_state_dict(optimizer_settings)

    unoptimized_model = None
    # compile the model
    if config.project.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if config.ddp:
        model = DDP(model, device_ids=[config.ddp.local_rank])

    return ModelTrainer(
        model=model,   # type: ignore
        model_args=model_args,
        scaler=scaler,
        optimizer=optimizer,
        config=config,
        current_learning_rate=config.training.learning_rate,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        unoptimized_model=unoptimized_model,
    )

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(ctx, trainer: ModelTrainer):
    out = {}
    trainer.model.eval()
    opts: list[Literal["train", "val"]] = ["train", "val"]
    for split in opts:
        losses = torch.zeros(trainer.config.output.eval_iters)
        for k in range(trainer.config.output.eval_iters):
            X, Y = get_batch(split, trainer.config)
            with ctx:
                _, loss = trainer.model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    trainer.model.train()
    return out

def report_loss_and_checkpoint(ctx, trainer: ModelTrainer):
    is_master_process = not trainer.config.ddp or trainer.config.ddp.master_process
    if trainer.iter_num % trainer.config.output.eval_interval == 0 and is_master_process:
        losses = estimate_loss(ctx, trainer)
        print(f"step {trainer.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if trainer.config.wandb.enabled:
            wandb.log({   # noqa
                "iter": trainer.iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": trainer.current_learning_rate,
                "mfu": trainer.running_mfu*100, # convert to percentage
            })
        if losses['val'] < trainer.best_val_loss or trainer.config.output.always_checkpoint:
            trainer.best_val_loss = losses['val']
            if trainer.iter_num > 0:
                checkpoint = {
                    'model': trainer.raw_model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'model_args': trainer.model_args,
                    'iter_num': trainer.iter_num,
                    'best_val_loss': trainer.best_val_loss,
                    'config': trainer.config,
                }
                print(f"saving checkpoint to {trainer.config.output_dir}")
                torch.save(checkpoint, os.path.join(trainer.config.output_dir, 'ckpt.pt'))

def log_timing(trainer: ModelTrainer, loss, start_time: float, local_iter_num: int) -> float:
    now = time.time()
    dt = now - start_time
    if trainer.iter_num % trainer.config.output.log_interval == 0:
        if not trainer.config.ddp or trainer.config.ddp.master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * trainer.config.training.gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = trainer.raw_model.estimate_mfu(trainer.config.training.batch_size * trainer.config.training.gradient_accumulation_steps, dt)
                trainer.running_mfu = mfu if trainer.running_mfu == -1.0 else 0.9*trainer.running_mfu + 0.1*mfu
            print(f"iter {trainer.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {trainer.running_mfu*100:.2f}%")
    return now

def training_loop(ctx, trainer: ModelTrainer):
    config = trainer.config
    model = trainer.model
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    X, Y = get_batch('train', config) # fetch the very first batch
    last_timestamp = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    loss = None
    while True:

        # determine and set the learning rate for this iteration
        lr = config.training.get_lr(trainer.iter_num) if config.training.decay.enabled else config.training.learning_rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        report_loss_and_checkpoint(ctx, trainer)
        if trainer.iter_num == 0 and config.project.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if config.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                trainer.model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)  # type: ignore
            with ctx:
                _, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train', config)
            # backward pass, with gradient scaling if training in fp16
            trainer.scaler.scale(loss).backward()
        # clip the gradient
        if config.training.adamw.grad_clip != 0.0:
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.adamw.grad_clip,
            )
        # step the optimizer and scaler if training in fp16
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        trainer.optimizer.zero_grad(set_to_none=True)

        # timing and logging
        last_timestamp = log_timing(trainer, loss, last_timestamp, local_iter_num)
        trainer.iter_num += 1
        local_iter_num += 1

        # termination conditions
        if trainer.iter_num > config.training.max_iters:
            break


def train_main(init_from: Literal["scratch", "resume"], config: NanoGPTConfig):
    ddp = get_ddp_config(config)
    config.ddp = ddp
    world_size = ddp.world_size if ddp else 1
    tokens_per_iter = config.training.gradient_accumulation_steps * world_size * config.training.batch_size * config.model.context_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if not ddp or ddp.master_process:
        os.makedirs(config.output_dir, exist_ok=True)

    seed_offset = ddp.seed_offset if ddp else 0
    torch.manual_seed(1337 + seed_offset)

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if config.device_type == 'cpu' else torch.amp.autocast(device_type=config.device_type, dtype=ptdtype)
    ####
    # logging
    if config.wandb.enabled:
        if not ddp or ddp.master_process:
            import wandb
            # TODO: push the config variables that are important along
            wandb.init(project=config.wandb.project_name, name=config.wandb.run_name)
    t = init_model(init_from, config)
    training_loop(ctx, t)

    ####
    if ddp:
        destroy_process_group()
