def config_sampler_args(parser):
    parser.add_argument("--num_cycles", type=int, default=500)
    parser.add_argument("--use_big", action="store_true")
    parser.add_argument("--step_size", type=float, default=1.5)
    parser.add_argument("--initial_balancing_constant", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=None)
    return parser


def config_acs_args(parser):
    parser.add_argument("--adapt_every", type=int, default=50)
    parser.add_argument("--burnin_budget", type=int, default=200)
    parser.add_argument("--burnin_adaptive", action="store_true")
    parser.add_argument("--burnin_test_steps", type=int, default=1)
    parser.add_argument("--step_obj", type=str, default="alpha_max")
    parser.add_argument("--burnin_init_bal", type=float, default=0.95)
    parser.add_argument("--a_s_cut", type=float, default=0.5)
    parser.add_argument("--burnin_lr", type=float, default=0.5)
    parser.add_argument("--burnin_error_margin_a_s", type=float, default=0.01)
    parser.add_argument("--burnin_error_margin_hops", type=float, default=5)
    parser.add_argument("--burnin_alphamin_decay", type=float, default=0.9)
    parser.add_argument("--bal_resolution", type=int, default=6)
    parser.add_argument("--burnin_step_obj", type=str, default="alphamax")
    parser.add_argument("--adapt_strat", type=str, default="greedy")
    parser.add_argument("--pair_optim", action="store_true")
    parser.add_argument("--kappa", type=float, default=1)
    parser.add_argument("--continual_adapt_budget", type=int, default=40)
    return parser


def config_acs_pcd_args(parser):
    parser.add_argument("--big_step", type=float, default=2.0)
    parser.add_argument("--use_manual_EE", action="store_true")
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
