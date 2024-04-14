from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import msgpack
import torch
from torch import Tensor
import os
import numpy as np

from nanogpt.config import NanoGPTConfig

@dataclass
class ContextConfig:
    default_device: str
    batch_size: int
    context_size: int
    data_dir: str

    @property
    def device_type(self):
        return "cuda" if "cuda" in self.default_device else "cpu"


class DataLoader(ABC):
    @abstractmethod
    def __init__(self, context_config: ContextConfig, config: dict[str, Any]):
        pass

    @abstractmethod
    def get_train_batch(self) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def get_val_batch(self) -> tuple[Tensor, Tensor]:
        pass


class FixedEntryDataLoader(DataLoader):
    def __init__(self, context_config: ContextConfig, config: dict[str, Any]):
        self.config = context_config
        print(f"{config=}")
        self.train_path = os.path.join(context_config.data_dir, config["train_path"])
        self.val_path = os.path.join(context_config.data_dir, config["val_path"])
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.record_size = self.input_size + self.output_size

    def get_train_batch(self) -> tuple[Tensor, Tensor]:
        data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        return self._get_batch(data)

    def get_val_batch(self) -> tuple[Tensor, Tensor]:
        data = np.memmap(self.val_path, dtype=np.uint16, mode='r')
        return self._get_batch(data)

    def _get_batch(self, data) -> tuple[Tensor, Tensor]:
        batch_size = self.config.batch_size
        entries = len(data) // self.record_size
        ix = torch.randint(entries, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i*self.record_size:i*self.record_size+self.input_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i*self.record_size + self.input_size:i*self.record_size + self.input_size + self.output_size]).astype(np.int64)) for i in ix])
        if self.config.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config.default_device, non_blocking=True), y.pin_memory().to(self.config.default_device, non_blocking=True)
        else:
            x, y = x.to(self.config.default_device), y.to(self.config.default_device)
        return x, y


class TokenStringDataLoader(DataLoader):
    def __init__(self, context_config: ContextConfig, config: dict[str, Any]):
        self.config = context_config
        self.train_path = os.path.join(context_config.data_dir, config["train_path"])
        self.val_path = os.path.join(context_config.data_dir, config["val_path"])

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    def get_train_batch(self) -> tuple[Tensor, Tensor]:
        data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        return self._get_batch(data)

    def get_val_batch(self) -> tuple[Tensor, Tensor]:
        data = np.memmap(self.val_path, dtype=np.uint16, mode='r')
        return self._get_batch(data)

    def _get_batch(self, data) -> tuple[Tensor, Tensor]:
        batch_size = self.config.batch_size
        block_size = self.config.context_size
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if self.config.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config.default_device, non_blocking=True), y.pin_memory().to(self.config.default_device, non_blocking=True)
        else:
            x, y = x.to(self.config.default_device), y.to(self.config.default_device)
        return x, y



def open_dataset(config: NanoGPTConfig) -> DataLoader:
    cc = ContextConfig(
        default_device=config.device,
        batch_size=config.training.batch_size,
        context_size=config.model.context_size,
        data_dir=config.data_dir,
    )
    dspath = os.path.join(config.data_dir, config.project.dataset_meta_file)
    if not os.path.isfile(dspath):
        d = {
            "train_path": "train.bin",
            "val_path": "val.bin",
        }
        return TokenStringDataLoader(cc, d)
    with open(dspath, "rb") as f:
        d = msgpack.load(f)
    assert isinstance(d, dict)
    match d["kind"]:
        case "token_string":
            return TokenStringDataLoader(cc, d)
        case "fixed_entry":
            return FixedEntryDataLoader(cc, d)
    raise Exception(f"Unknown dataset kind {d['kind']}")
