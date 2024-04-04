"""
Sample from a trained model
"""
import os
import pickle
import torch
import tiktoken
from config import NanoGPTConfig
from setup_context import setup_context
from model import GPTConfig, GPT
from dataclasses import dataclass

@dataclass
class ModelSampler:
    config: NanoGPTConfig
    num_samples: int = 10
    max_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200


def sample_main(sampler: ModelSampler, prompt: str | None = None, seed:int = 1337):
    if prompt is None:
        prompt = "\n"
    config = sampler.config
# start is the prompt variable
    ctx = setup_context(seed, config.device_type)
    ckpt_path = os.path.join(config.output_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(config.device)
    if config.project.compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    meta_path = os.path.join(config.data_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(sampler.num_samples):
                y = model.generate(x, sampler.max_tokens, temperature=sampler.temperature, top_k=sampler.top_k)
                print(decode(y[0].tolist()))
                print('---------------')
