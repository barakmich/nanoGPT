from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
import msgpack
import torch
from torch import Tensor
import os
import numpy as np

from nanogpt.config import NanoGPTConfig
from nanogpt.vocabs import VocabPair

@dataclass
class ContextConfig:
    default_device: str
    batch_size: int
    context_size: int
    data_dir: str
    vocab: VocabPair

    @property
    def device_type(self):
        return "cuda" if "cuda" in self.default_device else "cpu"


class BatchType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    OUTPUT_MASK = "output_mask"


class DataLoader(ABC):
    @abstractmethod
    def __init__(self, context_config: ContextConfig, config: dict[str, Any]):
        pass

    @abstractmethod
    def get_train_batch(self, get: list[BatchType]) -> list[Tensor]:
        pass

    @abstractmethod
    def get_val_batch(self, get: list[BatchType]) -> list[Tensor]:
        pass


class FixedEntryDataLoader(DataLoader):
    def __init__(self, context_config: ContextConfig, config: dict[str, Any]):
        self.config = context_config
        print(f"{config=}")
        self.train_path = os.path.join(context_config.data_dir, config["train_path"])
        self.val_path = os.path.join(context_config.data_dir, config["val_path"])
        self.offset_map = {}
        self.index_set = set()
        start = 0
        for f in config["fields"]:
            name = f["kind"]
            size = f["size"]
            self.offset_map[name] = (start, start + size)
            if f["indexes"]:
                self.index_set.add(name)
            start += size
        self.record_size = start

    def get_train_batch(self, get: list[BatchType]) -> list[Tensor]:
        data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        return self._get_batch(data, get)

    def get_val_batch(self, get: list[BatchType]) -> list[Tensor]:
        data = np.memmap(self.val_path, dtype=np.uint16, mode='r')
        return self._get_batch(data, get)

    def _get_batch(self, data, get: list[BatchType]) -> list[Tensor]:
        batch_size = self.config.batch_size
        entries = len(data) // self.record_size
        ix = torch.randint(entries, (batch_size,))
        record_offsets = [i * self.record_size for i in ix]
        out = []
        for g in get:
            start,end = self.offset_map[g.value]
            nps = [data[i+start:i+end] for i in record_offsets]
            if g.value in self.index_set:
                countvecs = [np.eye(self.config.vocab.output.vocab_size, dtype=np.float32)[n].sum(0).squeeze() for n in nps]
                for f in countvecs:
                    f[f != 0] = True
                    f[self.config.vocab.output.null_token_id] = False
                t = torch.stack([torch.from_numpy(n.astype(np.float16)) for n in countvecs])
                t = torch.unsqueeze(t, 1)
            else:
                t = torch.stack([torch.from_numpy((n).astype(np.int64)) for n in nps])
            if self.config.device_type == 'cuda' and g.value not in self.index_set:
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                t = t.pin_memory().to(self.config.default_device, non_blocking=True)
            else:
                t = t.to(self.config.default_device)
            out.append(t)
        return out


class TokenStringDataLoader(DataLoader):
    def __init__(self, context_config: ContextConfig, config: dict[str, Any]):
        self.config = context_config
        self.train_path = os.path.join(context_config.data_dir, config["train_path"])
        self.val_path = os.path.join(context_config.data_dir, config["val_path"])

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    def get_train_batch(self, get: list[BatchType]) -> list[Tensor]:
        data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        return self._get_batch(data, get)

    def get_val_batch(self, get: list[BatchType]) -> list[Tensor]:
        data = np.memmap(self.val_path, dtype=np.uint16, mode='r')
        return self._get_batch(data, get)

    def _get_batch(self, data, get: list[BatchType]) -> list[Tensor]:
        batch_size = self.config.batch_size
        block_size = self.config.context_size
        ix = torch.randint(len(data) - block_size, (batch_size,))
        out = []
        for g in get:
            if g == BatchType.INPUT:
                x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            elif g == BatchType.OUTPUT:
                x = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            else:
                raise Exception("TokenStringDataLoader does not support output masking")
            if self.config.device_type == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x = x.pin_memory().to(self.config.default_device, non_blocking=True)
            else:
                x = x.to(self.config.default_device)
            out.append(x)
        return out



def open_dataset(config: NanoGPTConfig) -> DataLoader:
    cc = ContextConfig(
        default_device=config.device,
        batch_size=config.training.batch_size,
        context_size=config.model.context_size,
        data_dir=config.data_dir,
        vocab=config.vocab,
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
