from typing import Any
import tiktoken
import msgpack
from abc import ABC, abstractmethod, abstractproperty

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

class TikTokenVocabulary(Vocabulary):
    kind: str = "tiktoken"

    def __init__(self, name: str, *, allowed_special: set[str] | None = None):
        self.name = name
        self.encoding = tiktoken.get_encoding(name)
        self.allowed_special = allowed_special if allowed_special is not None else set()

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
        return self.encoding.max_token_value - 1

    def to_dict(self):
        return {
            "kind": self.kind,
            "name": self.name,
            "allowed_special": list(self.allowed_special),
        }

class DictVocabulary(Vocabulary):
    kind: str = "dict"

    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi[x] for x in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.itos[i] for i in indices]

    def encode_string(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode_string(self, tokens: list[int]) -> str:
        return ''.join(self.decode(tokens))

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "stoi": self.stoi,
        }


class VocabPair:
    def __init__(self, input: Vocabulary, output: Vocabulary | None) -> None:
        self.input = input
        self.output = output

    def _build_dict(self) -> dict[str, Any]:
        return {
            "input": self.input.to_dict(),
            "output": self.output.to_dict() if self.output else None,
        }

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
        if d["output"] is not None:
            output = instantiate_vocab(d["output"])
        return VocabPair(input=input, output=output)


def instantiate_vocab(d: dict) -> Vocabulary:
    match d["kind"]:
        case "tiktoken":
            return TikTokenVocabulary(
                name=d["name"],
                allowed_special=set(d["allowed_special"]),
            )
        case "dict":
            return DictVocabulary(d["stoi"])
        case _:
            raise Exception("Unexpected input dictionary kind")
