from libai.config import default_argument_parser


def remat_argument_parser(epilog=None):
    parser = default_argument_parser(epilog=epilog)
    parser.add_argument("--threshold", type=int)
    return parser
