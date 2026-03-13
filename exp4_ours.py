import argparse

from attribute_experiments import add_common_args, bootstrap_common, run_experiment_exp4


def build_parser():
    parser = argparse.ArgumentParser("Exp-4 Ours: dual path + merge + attribute reranking")
    add_common_args(parser)
    parser.add_argument("--disable_merge", action="store_true", help="skip candidate fusion deduplication")
    parser.add_argument("--disable_rerank", action="store_true", help="disable attribute-aware reranking")
    parser.add_argument("--subject_prompt_override", type=str, default="", help="override subject prompt")
    parser.add_argument("--attr_prompt_override", type=str, default="", help="override attribute prompt")
    return parser


def main():
    args = build_parser().parse_args()
    context = bootstrap_common(args)
    run_experiment_exp4(context, args)


if __name__ == "__main__":
    main()
