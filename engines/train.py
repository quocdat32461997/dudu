import argparse

from engines.utils import TrainerFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fn", type=str, required=True)
    parser.add_argument("--configs", type=str, required=True)

    args = parser.parse_args()

    train_fn = TrainerFactory.get(args.train_fn)
    train_fn(args.configs)
