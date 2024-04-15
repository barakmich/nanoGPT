from nanogpt.config import NanoGPTConfig
from nanogpt.data_loader import BatchType, open_dataset


def head_main(config: NanoGPTConfig):
    data_loader = open_dataset(config)
    vocab = config.vocab
    x, y = tuple(data_loader.get_train_batch([BatchType.INPUT, BatchType.OUTPUT]))
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    for i in range(x.shape[0]):
        print("******")
        print(f"Input: {x[i]}")
        print(f"Output: {y[i]}")
        intoks = vocab.input.decode(x[i].tolist())
        print(f"InToks: {intoks}")
        outtoks = vocab.output.decode(y[i].tolist())
        print(f"OutToks: {outtoks}")



