import argparse

from attribute_experiments import add_common_args, bootstrap_common, run_experiment_exp3


def build_parser():
    parser = argparse.ArgumentParser("Exp-3 Attr-only: attribute prompt recall + attribute reranking")
    add_common_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    context = bootstrap_common(args)
    run_experiment_exp3(context, args)


if __name__ == "__main__":
    main()
