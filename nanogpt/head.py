from nanogpt.config import NanoGPTConfig
from nanogpt.data_loader import BatchType, open_dataset


def head_main(config: NanoGPTConfig):
    data_loader = open_dataset(config)
    vocab = config.vocab
    for t in [BatchType.INPUT, BatchType.OUTPUT, BatchType.OUTPUT_MASK]:
        print("******")
        try:
            l = data_loader.get_train_batch([t])
            x = l[0]
        except Exception as e:
            print(f"{e.args}\n")
            continue
        print(f"{t.value} shape: {x[0].shape}")
        for i in range(x.shape[0]):
            print(f"{t.value}: {x[i]}")
            if t == BatchType.INPUT:
                intoks = vocab.input.decode(x[i].tolist())
                print(f"{t.value} InToks: {intoks}")
            elif t == BatchType.OUTPUT_MASK:
                print(f"Mask Total: {sum(x[i].tolist())} {x[i]}")
            else:
                outtoks = vocab.output.decode(x[i].tolist())
                print(f"{t.value} OutToks: {outtoks}")



