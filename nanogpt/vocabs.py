from typing import Any
import tiktoken
import msgpack
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum

def optimize_vocab_size(i: int) -> int:
    # Bring it to the next multiple of 64, for performance reasons.
    # TODO: Does it need to be above a threshhold to see the usefulness of it?
    off = i % 64
    if off == 0:
        return i
    return (64 - off) + i


class SpecialToken(Enum):
    NULL = "<|nul|>"
    END = "<|end|>"


class Vocabulary(ABC):
    kind: str

    @abstractmethod
    def encode_string(self, s: str) -> list[int]:
        pass

    @abstractmethod
    def encode(self, tokens: list[str]):
        pass

    @abstractmethod
    def decode(self, idxs: list[int]) -> list[bytes]:
        pass

    @abstractmethod
    def decode_string(self, idx: list[int]) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractproperty
    def vocab_size(self) -> int:
        return 0

    def null_token_id(self) -> int | None:
        return None

    @abstractmethod
    def set_optimize(self, optimize: bool):
        pass


class TikTokenVocabulary(Vocabulary):
    kind: str = "tiktoken"

    def __init__(self, name: str, *, allowed_special: set[str] | None = None):
        self.name = name
        self.encoding = tiktoken.get_encoding(name)
        self.allowed_special = allowed_special if allowed_special is not None else set()
        self.optimize = False

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.encoding.encode_single_token(x) for x in tokens]

    def encode_string(self, s: str):
        return self.encoding.encode(s, allowed_special=self.allowed_special)

    def decode(self, idxs: list[int]) -> list[bytes]:
        return [self.encoding.decode_single_token_bytes(x) for x in idxs]

    def decode_string(self, idxs: list[int]) -> str:
        return self.encoding.decode(idxs)

    @property
    def vocab_size(self) -> int:
        v = self.encoding.max_token_value - 1
        return optimize_vocab_size(v) if self.optimize else v

    def set_optimize(self, optimize: bool):
        self.optimize = optimize

    def to_dict(self):
        return {
            "kind": self.kind,
            "name": self.name,
            "allowed_special": list(self.allowed_special),
        }

class DictVocabulary(Vocabulary):
    kind: str = "dict"

    def __init__(self, stoi: dict[str, int], sep: str = ""):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.sep = sep
        self.optimize = False

    def encode(self, tokens: list[str]) -> list[int]:
        if self.null_token_id is not None:
            return [self.stoi[c] if c != '' else self.null_token_id for c in tokens]
        return [self.stoi[x] for x in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.itos[i] for i in indices]

    def encode_string(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]


    def decode_string(self, tokens: list[int]) -> str:
        return self.sep.join(self.decode(tokens))

    @property
    def null_token_id(self) -> int | None:
        return self.stoi.get(SpecialToken.NULL.value, None)

    def set_optimize(self, optimize: bool):
        self.optimize = optimize

    @property
    def vocab_size(self) -> int:
        v = len(self.stoi)
        return optimize_vocab_size(v) if self.optimize else v

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "stoi": self.stoi,
            "sep": self.sep,
            "optimize": self.optimize,
        }


class VocabPair:
    def __init__(self, input: Vocabulary, output: Vocabulary | None) -> None:
        self.input = input
        self._output = output

    def _build_dict(self) -> dict[str, Any]:
        return {
            "input": self.input.to_dict(),
            "output": self._output.to_dict() if self._output else None,
        }

    def set_optimize(self, optimize: bool):
        self.input.set_optimize(optimize)
        if self._output is not None:
            self.output.set_optimize(optimize)

    def dump(self, writer: Any):
        return msgpack.dump(self._build_dict(), writer)

    def dumps(self):
        return msgpack.dumps(self._build_dict())

    @staticmethod
    def load(reader: Any) -> "VocabPair":
        return VocabPair._load_pair(msgpack.load(reader))

    @staticmethod
    def loads(s: str) -> "VocabPair":
        return VocabPair._load_pair(msgpack.loads(s))

    @staticmethod
    def _load_pair(d) -> "VocabPair":
        assert isinstance(d, dict)
        input = instantiate_vocab(d["input"])
        output = input
        if "output" in d and d["output"] is not None:
            output = instantiate_vocab(d["output"])
        return VocabPair(input=input, output=output)

    @property
    def output(self) -> Vocabulary:
        return self._output if self._output is not None else self.input


def instantiate_vocab(d: dict) -> Vocabulary:
    match d["kind"]:
        case "tiktoken":
            return TikTokenVocabulary(
                name=d["name"],
                allowed_special=set(d["allowed_special"]),
            )
        case "dict":
            return DictVocabulary(
                d["stoi"],
                sep=d["sep"] if "sep" in d else "",
            )
        case _:
            raise Exception("Unexpected input dictionary kind")
