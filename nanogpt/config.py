import tomllib
import math
import os
from dataclasses import dataclass
from pathlib import Path

from nanogpt.vocabs import VocabPair

@dataclass
class ProjectConfig:
    name: str
    dataset_dir: str
    compile: bool = True
    device: str = "cuda"  # "cpu", "cuda", "cuda:0", "mps" for Macs
    eval_only: bool = False
    ddp_backend: str = "nccl"  # "nccl", "gloo", etc
    dataset_meta_file: str = "dataset.msgpack"

@dataclass
class OutputConfig:
    out_dir: str
    eval_interval: int = 2000
    eval_iters: int = 200
    log_interval: int = 1
    always_checkpoint: bool = True

@dataclass
class WandbConfig:
    enabled: bool = False
    project_name: str = ""
    run_name: str = ""

@dataclass
class ModelConfig:
    layers: int
    heads: int
    embedding_dimension: int
    context_size: int
    optimize_vocab_size: bool = True
    bias: bool = False
    causal: bool = True
    avg_logits: bool = False
    output_mask: bool = False
    final_norm: bool = False
    positional_embeddings: bool = True
    null_token_mask: bool = False

@dataclass
class AdamWConfig:
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

@dataclass
class LRDecayConfig:
    enabled: bool = True
    warmup_iters: int  = 2000
    decay_iters: int | None = None  # could be equal to max_iters
    min_learning_rate: float | None = None  # could be equal to learning_rate / 10


@dataclass
class TrainingConfig:
    adamw: AdamWConfig
    decay: LRDecayConfig
    batch_size: int
    gradient_accumulation_steps: int
    dropout: float = 0.0
    max_iters: int = 600000

    @property
    def learning_rate(self) -> float:
        return self.adamw.learning_rate

    def get_lr(self, iteration: int) -> float:
        """learning rate decay scheduler (cosine with warmup)"""
        assert self.decay.decay_iters is not None
        assert self.decay.min_learning_rate is not None
        # 1) linear warmup for warmup_iters steps
        if iteration < self.decay.warmup_iters:
            return self.adamw.learning_rate * iteration / self.decay.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iteration > self.decay.decay_iters:
            return self.decay.min_learning_rate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - self.decay.warmup_iters) / (self.decay.decay_iters - self.decay.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.decay.min_learning_rate + coeff * (self.adamw.learning_rate - self.decay.min_learning_rate)

@dataclass
class DDPConfig:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 0

    @property
    def master_process(self):
        return self.rank == 0

    @property
    def seed_offset(self):
        return self.rank



@dataclass
class NanoGPTConfig:
    project: ProjectConfig
    output: OutputConfig
    wandb: WandbConfig
    model: ModelConfig
    training: TrainingConfig
    base_path: str
    ddp: DDPConfig | None = None
    _vocab: VocabPair | None = None

    @property
    def vocab(self) -> VocabPair:
        if self._vocab is not None:
            return self._vocab
        vocab_path = os.path.join(self.data_dir, "vocab.msgpack")
        with open(vocab_path, "rb") as f:
            self._vocab = VocabPair.load(f)
        return self._vocab

    @property
    def output_dir(self) -> str:
        return os.path.join(self.base_path, self.output.out_dir)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_path, self.project.dataset_dir)

    @property
    def device_type(self) -> str:
        return "cuda" if "cuda" in self.project.device else "cpu"

    @property
    def device(self) -> str:
        return self.project.device

    @staticmethod
    def from_toml(path: Path) -> "NanoGPTConfig":
        with open(path, "rb") as f:
            d = tomllib.load(f)
        lrd = LRDecayConfig(**d["training"]["lr_decay"])
        del d["training"]["lr_decay"]
        adamw = AdamWConfig(**d["training"]["adamw"])
        del d["training"]["adamw"]
        if lrd.min_learning_rate is None:
            lrd.min_learning_rate = adamw.learning_rate / 10.0
        train = TrainingConfig(adamw=adamw, decay=lrd, **d["training"])
        if train.decay.decay_iters is None:
            train.decay.decay_iters = train.max_iters
        model = ModelConfig(**d["model"])
        output = OutputConfig(**d["output"])
        project = ProjectConfig(**d["project"])
        if "wandb" in d:
            wandb = WandbConfig(**d["wandb"])
        else:
            wandb = WandbConfig()
        conf = NanoGPTConfig(project=project, output=output,model=model, training=train, wandb=wandb, base_path=os.path.abspath(os.path.dirname(path)))
        return conf

