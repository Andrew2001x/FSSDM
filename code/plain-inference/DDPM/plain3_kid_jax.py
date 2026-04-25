import plain_kid_jax as kid
from flax_ddpm.script_utils import get_args


def get_parser():
    parser = kid.get_parser()
    for action in parser._actions:
        if action.dest == "variant":
            action.required = False
            action.default = "plain3jax"
    return parser


def main(args):
    args.variant = "plain3jax"
    kid.main(args)


if __name__ == "__main__":
    main(get_args(get_parser()))
