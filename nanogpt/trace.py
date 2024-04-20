"""
Dump a trained model (for other libtorch purposes)
"""
import os
import torch
import random
from nanogpt.config import NanoGPTConfig
from nanogpt.vocabs import VocabPair
from nanogpt.setup_context import setup_context
from nanogpt.model import GPTConfig, GPT
from dataclasses import dataclass

@dataclass
class ModelSampler:
    config: NanoGPTConfig
    num_samples: int = 10
    max_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200


def trace_main(
        config: NanoGPTConfig,
        output_filename: str = "export.pt",
        output_device = None,
    ):
    device = config.device if output_device is None else output_device
    ckpt_path = os.path.join(config.output_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    # if config.project.compile:
        # model = torch.compile(model) # requires PyTorch 2.0 (optional)

    vocab_path = os.path.join(config.data_dir, 'vocab.msgpack')
    with open(vocab_path, "rb") as f:
        vocab = VocabPair.load(f)
    vocab.set_optimize(True)
    print(f"Input vocab size: {vocab.input.vocab_size}, Output: {vocab.output.vocab_size}")
    print(f"Model output size: {model.config.output_vocab_size}")

    start_ids =  []
    for _ in range(config.model.context_size):
        n = random.randint(0, vocab.input.vocab_size - 1)
        start_ids.append(n)
    x = (torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...])

    print(f"Input shape: {x.shape}")
    inputs = {"embeddings_and_logits": x}
    traced = torch.jit.trace_module(model, inputs=inputs)
    traced.save(os.path.join(config.output_dir, output_filename))
    # output = model.embeddings_and_logits(x)
    # print(f"Output shape: {output[0].shape} {output[1].shape}")
