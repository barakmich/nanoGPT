import argparse
from pathlib import Path
import logging

from config import NanoGPTConfig
from sample import ModelSampler, sample_main
from train import train_main

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(prog="nanogpt")
    parser.add_argument("-c", "--config", help="Config file")
    subparsers = parser.add_subparsers(title="subcommands")

    train_parser = subparsers.add_parser("train", help="train a model")
    train_parser.add_argument("--eval_iters", type=int)
    train_parser.add_argument("-i", "--iters", type=int)
    train_parser.add_argument("--resume", action="store_true", default=False, help="Resume from last checkpoint")
    train_parser.set_defaults(subcommand="train")

    sample_parser = subparsers.add_parser("sample", help="sample from a model")
    sample_parser.add_argument("-p", "--prompt", help="Starting prompt")
    sample_parser.add_argument("-n", "--num_samples", type=int, help="Number of turns to sample")
    sample_parser.add_argument("--max_tokens", type=int, help="Return up to this many tokens in a turn")
    sample_parser.add_argument("--temperature", type=float)
    sample_parser.add_argument("--top_k", type=int, help="Number of top tokens to sample from (generate nothing too weird)")
    sample_parser.set_defaults(subcommand="sample")

    config_parser = subparsers.add_parser("print-config", help="train a model")
    config_parser.set_defaults(subcommand="config")


    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.fatal(f"Config file at {config_path} does not exist")
    logger.info(f"Using config file {config_path}")

    ngpt_config = NanoGPTConfig.from_toml(config_path)

    if args.subcommand == "train":
        if args.resume:
            train_main("resume", ngpt_config)
        else:
            train_main("scratch", ngpt_config)
    elif args.subcommand == "sample":
        sampler = ModelSampler(config=ngpt_config)
        if args.num_samples:
            sampler.num_samples = args.num_samples
        if args.max_tokens:
            sampler.max_tokens = args.max_tokens
        if args.temperature:
            sampler.temperature = args.temperature
        if args.top_k:
            sampler.top_k = args.top_k
        sample_main(sampler, args.prompt)

    elif args.subcommand == "config":
        print(str(ngpt_config))
    else:
        logger.fatal("No such subcommand")

if __name__ == '__main__':
    main()
