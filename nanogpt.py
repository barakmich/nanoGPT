import argparse
from pathlib import Path
import logging

from config import NanoGPTConfig

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(prog="nanogpt")
    parser.add_argument("-c", "--config", help="Config file")
    subparsers = parser.add_subparsers(title="subcommands")
    train_parser = subparsers.add_parser("train", help="train a model")
    train_parser.add_argument("--eval_iters", type=int)
    train_parser.add_argument("-i", "--iters", type=int)
    train_parser.set_defaults(subcommand="train")
    sample_parser = subparsers.add_parser("sample", help="sample from a model")
    sample_parser.add_argument("-p", "--prompt", help="Starting prompt")
    sample_parser.set_defaults(subcommand="sample")
    config_parser = subparsers.add_parser("print-config", help="train a model")
    config_parser.set_defaults(subcommand="config")
    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.fatal(f"Config file at {config_path} does not exist")
    logger.info(f"Using config file {config_path}")

    ngpt_config = NanoGPTConfig.from_toml(config_path)
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

    if args.subcommand == "train":
        pass
    elif args.subcommand == "sample":
        pass
    elif args.subcommand == "config":
        print(str(ngpt_config))
    else:
        logger.fatal("No such subcommand")

if __name__ == '__main__':
    main()
