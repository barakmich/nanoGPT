
# nanoGPT (Barak's Version)

This is a personal fork of [nanoGPT](https://github.com/karpathy/nanoGPT) where I endeavor to make nanoGPT useful to myself, by taking it apart and putting it back together again.

Understandably, the original is geared toward the pedagogy of it all -- and it's very good at that!
So my experiments here are making it interact well with my greater aesthetic.

Big changes include:

- [X] Use [rye](https://rye-up.com) as the venv/package manager.
- [X] No `eval()` based configuration -- use a real TOML config for each dataset/model
- [X] Add a richer command-line tool, read real flags, and set up the configurations and then call train/sample
- [X] Make actual configuration structs, and remove global variables as much as possible.
  - [X] This includes making real functions, adding type signatures, etc, instead of just a one-shot Python run-through
- [X] Fewer implicit assumptions
  - [X] Vocabularies are well-defined and stored in MsgPack, not `meta.pkl` Pickle (so as to generate datasets with other languages!)
  - [ ] Datasets, similarly, should not be Numpy dumps, but a cross-language storage format that also considers input/target training (probably Parquet)

Still (always) in progress and no outside contributions are ever to be accepted.
This one's just for me.
But I encourage you to fork and/or build your own.

## Getting Started

First, go read the [original README](https://github.com/karpathy/nanoGPT).
Then, the quickstart here will make more sense:

### Set up the venv

Start by installing `rye` (or your favorite venv manager). 
This assumes you're inside the venv (or, equivalently, prefix things with `rye run`).
So, eg `source activate.sh`

### Set up the dataset
```
(nanogpt) $ cd shakespeare_char/data; python prepare.py; cd ../..
```

### Tweak the config to your liking

Edit `shakespeare_char/config.toml` for your hyperparameters or local setup (eg, set `"mps"` as the training `device` for Macs)

### Train based on the config
```
(nanogpt) $ python nanogpt.py -c shakespeare_char/config.toml train
```

### Sample based on the config
```
(nanogpt) $ python nanogpt.py -c shakespeare_char/config.toml sample -p "JULIET: Thy" -n 3
```
---
Now go forth and create your own configs, export datasets (with Python or not) and train your own nanoGPTs!
