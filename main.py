import argparse
from runner import Runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False, help='train or evaluate, default eval')
    args = parser.parse_args()
    runner = Runner()
    if args.train:
        runner.train()
    else:
        runner.eval()
