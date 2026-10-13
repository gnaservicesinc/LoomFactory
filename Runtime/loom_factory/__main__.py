import argparse
import sys

from .engine import run_cli


def main():
    parser = argparse.ArgumentParser(prog="loom_factory")
    parser.add_argument("command", choices=("train-map",))
    parser.add_argument("--request", required=True, help="Project-owned training request JSON")
    args = parser.parse_args()
    return run_cli(args.command, args.request)


if __name__ == "__main__":
    sys.exit(main())
