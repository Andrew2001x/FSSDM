import plain_ssim_jax as ssim


def get_parser():
    parser = ssim.get_parser()
    for action in parser._actions:
        if action.dest == "variant":
            action.required = False
            action.default = "plain1jax"
    return parser


def main(args):
    args.variant = "plain1jax"
    ssim.main(args)


if __name__ == "__main__":
    main(get_parser().parse_args())
