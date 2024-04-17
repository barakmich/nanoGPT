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
from typing import Literal

from dataclasses import dataclass

import torch
import torch.amp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim import AdamW
from torch.optim.optimizer import StateDict

from nanogpt.config import DDPConfig, NanoGPTConfig
from nanogpt.data_loader import BatchType, DataLoader, open_dataset
from nanogpt.setup_context import setup_context
from nanogpt.model import GPTConfig, GPT

# -----------------------------------------------------------------------------
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

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


@dataclass
class ModelTrainer:
    model: GPT | DDP
    model_args: dict
    scaler: GradScaler
    optimizer: AdamW
    config: NanoGPTConfig
    data_loader: DataLoader
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
    vocab = config.vocab

    vocab.set_optimize(config.model.optimize_vocab_size)
    in_vocab_size = vocab.input.vocab_size
    out_vocab_size = vocab.output.vocab_size

    model_args = dict(
        n_layer=config.model.layers,
        n_head=config.model.heads,
        n_embd=config.model.embedding_dimension,
        block_size=config.model.context_size,
        bias=config.model.bias,
        vocab_size=in_vocab_size,
        output_vocab_size=out_vocab_size,
        causal=config.model.causal,
        final_norm=config.model.final_norm,
        sum_logits=config.model.sum_logits,
        dropout=config.training.dropout,
        positional_embeddings=config.model.positional_embeddings,
        null_token=None if not config.model.null_token_mask else config.vocab.input.null_token_id,
    ) # start with model_args from command line

    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
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
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'output_vocab_size']:
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
        print("model compiled")

    # wrap model into DDP container
    if config.ddp:
        model = DDP(model, device_ids=[config.ddp.local_rank])

    data_loader = open_dataset(config)

    return ModelTrainer(
        model=model,   # type: ignore
        model_args=model_args,
        scaler=scaler,
        data_loader=data_loader,
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
            print(".", end="", flush=True)
            if split == "train":
                if trainer.config.model.output_mask:
                    X, Y, M = tuple(trainer.data_loader.get_train_batch([BatchType.INPUT, BatchType.OUTPUT, BatchType.OUTPUT_MASK]))
                else:
                    X, Y = tuple(trainer.data_loader.get_train_batch([BatchType.INPUT, BatchType.OUTPUT]))
                    M = None
            else:
                if trainer.config.model.output_mask:
                    X, Y, M = tuple(trainer.data_loader.get_val_batch([BatchType.INPUT, BatchType.OUTPUT, BatchType.OUTPUT_MASK]))
                else:
                    X, Y = tuple(trainer.data_loader.get_val_batch([BatchType.INPUT, BatchType.OUTPUT]))
                    M = None

            with ctx:
                _, loss = trainer.model(X, Y, output_masks=M)
            losses[k] = loss.item()
        print("")
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
    use_output_mask = trainer.config.model.output_mask
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    if use_output_mask:
        X, Y, M = tuple(trainer.data_loader.get_train_batch([BatchType.INPUT, BatchType.OUTPUT, BatchType.OUTPUT_MASK])) # fetch the very first batch
    else:
        X, Y = tuple(trainer.data_loader.get_train_batch([BatchType.INPUT, BatchType.OUTPUT])) # fetch the very first batch
        M = None

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
                _, loss = model(X, Y, output_masks=M)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            if use_output_mask:
                X, Y, M = tuple(trainer.data_loader.get_train_batch([BatchType.INPUT, BatchType.OUTPUT, BatchType.OUTPUT_MASK])) # fetch the very first batch
            else:
                X, Y = tuple(trainer.data_loader.get_train_batch([BatchType.INPUT, BatchType.OUTPUT]))
                M = None
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
    ctx = setup_context(1337 + seed_offset, config.device_type)

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
