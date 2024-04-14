from nanogpt.config import NanoGPTConfig
from nanogpt.data_loader import open_dataset
from nanogpt.train import get_vocab


def head_main(config: NanoGPTConfig):
    data_loader = open_dataset(config)
    vocab = get_vocab(config)
    x, y = data_loader.get_train_batch()
    print(f"X shape: {x.shape}")
    print(f"Y shape: {x.shape}")
    for i in range(x.shape[0]):
        print("******")
        print(f"Input: {x[i]}")
        print(f"Output: {y[i]}")
        intoks = vocab.input.decode(x[i].tolist())
        print(f"InToks: {intoks}")
        outtoks = vocab.output.decode(y[i].tolist())
        print(f"OutToks: {outtoks}")



