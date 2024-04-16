from nanogpt.config import NanoGPTConfig
from nanogpt.data_loader import BatchType, open_dataset


def head_main(config: NanoGPTConfig):
    config.training.batch_size = 2
    data_loader = open_dataset(config)
    vocab = config.vocab
    fetch = [BatchType.INPUT, BatchType.OUTPUT, BatchType.OUTPUT_MASK]
    try:
        l = data_loader.get_train_batch(fetch)
    except Exception as e:
        print(f"{e.args}\n")
        raise e
    for t, x in zip(fetch, l):
        print("******")
        print(f"{t.value} shape: {x.shape}")
        for i in range(x.shape[0]):
            print(f"{i}: {x[i]}")
            if t == BatchType.INPUT:
                intoks = vocab.input.decode(x[i].tolist())
                print(f"{t.value} InToks: {intoks}")
            elif t == BatchType.OUTPUT_MASK:
                #print(f"Mask Total: {sum(x[i].tolist())} {x[i]}")
                print(f"Output Mask")
            else:
                outtoks = vocab.output.decode(x[i].tolist())
                print(f"{t.value} OutToks: {outtoks}")



