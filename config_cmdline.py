def config_acs_args(parser):
    parser.add_argument("--num_cycles", type=int, default=250)
    parser.add_argument("--burnin_budget", type=int, default=200)
    parser.add_argument("--burnin_big_bal", type=float, default=.95)
    parser.add_argument("--burnin_small_bal", type=float, default=.5)
    parser.add_argument("--a_s_cut", type=float, default=0.5)
    parser.add_argument("--burnin_lr", type=float, default=0.5)
    parser.add_argument("--bal_resolution", type=int, default=10)
    parser.add_argument("--initial_balancing_constant", type=float, default=.5)
    return parser


def config_acs_pcd_args(parser):
    parser.add_argument("--big_step", type=float, default=2.0)
    parser.add_argument("--use_manual_EE", action="store_true")
    parser.add_argument("--adapt_every", type=int, default=25)
    parser.add_argument("--big_step_sampling_steps", type=int, default=5)
    parser.add_argument("--small_step", type=float, default=0.2)
    parser.add_argument("--small_bal", type=float, default=0.5)
    parser.add_argument("--big_bal", type=float, default=1.0)
    return parser


potential_datasets = [
    "mnist",
    "fashion",
    "emnist",
    "caltech",
    "omniglot",
    "kmnist",
    "random",
]
