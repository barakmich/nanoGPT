"""
Sample from a trained model
"""
import os
import torch
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


def sample_main(
        sampler: ModelSampler,
        prompt: str | list[str] | None = None,
        seed:int = 1337,
        last: bool = False,
        top_n_probs: int | None = None
    ):
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

    vocab_path = os.path.join(config.data_dir, 'vocab.msgpack')
    with open(vocab_path, "rb") as f:
        vocab = VocabPair.load(f)

    # encode the beginning of the prompt
    if isinstance(prompt, list):
        start_ids = vocab.input.encode(prompt)
    else:
        start_ids = vocab.input.encode_string(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(sampler.num_samples):
                y = model.generate(x, sampler.max_tokens, temperature=sampler.temperature, top_k=sampler.top_k, as_vec=top_n_probs is not None)
                if top_n_probs is not None:
                    tops = y[0].argsort(descending=True).tolist()[:top_n_probs]
                    toks = vocab.output.decode(tops)
                    for tok, idx in zip(toks, tops):
                        print(f"{tok}: {y[0][idx] * 100:.5f}%")
                    break

                else:
                    out_toks = y[0].tolist()
                    if last:
                        print(vocab.output.decode_string([out_toks[-1]]))
                    else:
                        print(vocab.output.decode_string(out_toks))
                print('---------------')
