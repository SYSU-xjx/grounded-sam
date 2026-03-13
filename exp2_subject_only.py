import argparse

from attribute_experiments import add_common_args, bootstrap_common, run_experiment_exp2


def build_parser():
    parser = argparse.ArgumentParser("Exp-2 Subject-only: category recall + attribute reranking")
    add_common_args(parser)
    parser.add_argument(
        "--attribute_only_rerank",
        action="store_true",
        help="set final score to ignore attribute score contribution and keep pure detection ranking",
    )
    return parser


def main():
    args = build_parser().parse_args()
    context = bootstrap_common(args)
    run_experiment_exp2(context, args)


if __name__ == "__main__":
    main()
