import argparse
from pathlib import Path

# from methods import METHODS
from data import DATASETS


def process_args(args):
    # >>> Check arguments' validation <<<

    if args.singlelinear_eval and args.multilinear_eval:
        raise NotImplementedError(
            "Single-layer and Multi-layer linear evaluation cannot be done together."
        )
    if not args.singlelinear_eval and not args.multilinear_eval and not args.cka_eval:
        print(
            "No evaluation setting was selected.\nSwitching to *Single-layer* Evaluation."
        )
        args.singlelinear_eval = 1

    if args.eval_layer < 1 or args.eval_layer > 4:
        args.eval_layer = 4

    if args.unsupervised and not args.keep_training_data:
        raise NotImplementedError(
            "'keep_training_data' should be 1 with '--unsupervised 1'. No continual evaluation method is implemented for unsupervised settings yet."
        )
    if args.unsupervised and not args.use_augs:
        raise NotImplementedError(
            "'use_augs' should be 1 with '--unsupervised 1'. Unsupervised methods need data augmentations."
        )
    if args.multilinear_eval and not args.keep_training_data:
        raise NotImplementedError(
            "'keep_training_data' should be 1 with '--multilinear_eval 1'. No implementation for continual multi-layer linear evaluation."
        )

    # Process lr_decay_epochs
    iterations = args.lr_decay_epochs.split(",")
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # Update logging directory
    run_details = f"{args.dataset}_U{args.unsupervised}_{args.method}_NT{args.n_tasks}_M{args.mem_size}_E{args.n_epochs}_{args.model}_NF{args.nf}_LR{args.lr}_TMP{args.supcon_temperature}_{args.run}/"
    # args.log_path += run_details
    args.snapshot_path += run_details

    # Create snapshots directory
    if args.save_snapshot:
        Path(args.snapshot_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving snapshots in {args.snapshot_path}")

    return args


def get_parser():
    # >>> Arguments <<<
    parser = argparse.ArgumentParser()

    """ optimization (fixed across all settings) """
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_batch_size", type=int, default=64)

    # choose your weapon
    parser.add_argument("-m", "--method", type=str, default="er")
    parser.add_argument("--model", type=str, default="resnet18")

    """ data """
    parser.add_argument("--download", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="../cl-datasets")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=DATASETS)
    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--probe_num_samples", type=int, default=-1)

    parser.add_argument("--iid_split", type=int, default=0)
    parser.add_argument("--half_iid", type=int, default=0)

    parser.add_argument("--nf", type=int, default=64)

    """ setting """
    parser.add_argument("--use_snapshots", type=int, default=0)
    parser.add_argument("--unsupervised", type=int, default=0, help="")
    parser.add_argument("--task_incremental", type=int, default=0)
    parser.add_argument(
        "--singlelinear_eval",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--multilinear_eval", type=int, default=0, help="Multi-layer linear evaluation"
    )
    parser.add_argument("--cka_eval", type=int, default=0, help="CKA analysis")
    parser.add_argument("--keep_training_data", type=int, default=0, help="")
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1e9,
    )
    parser.add_argument(
        "--eval_layer",
        type=int,
        default=-1,
    )
    parser.add_argument("--nme_classifier", type=int, default=0)
    parser.add_argument("--n_adaptation_epochs", type=int, default=0)

    parser.add_argument("--n_epochs", type=int, default=60)
    parser.add_argument("--n_warmup_epochs", type=int, default=0)
    parser.add_argument("--n_iters", type=int, default=1)
    parser.add_argument("--n_tasks", type=int, default=-1)
    parser.add_argument("--task_free", type=int, default=0)
    parser.add_argument("--use_augs", type=int, default=1)
    parser.add_argument("--samples_per_task", type=int, default=-1)
    parser.add_argument("--mem_size", type=int, default=0, help="controls buffer size")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--validation", type=int, default=0)
    parser.add_argument("--load_best_args", type=int, default=0)

    """ logging """
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_project", type=str, default="SCRD")
    parser.add_argument("--exp_name", type=str, default="tmp")

    parser.add_argument("--save_snapshot", type=int, default=0)
    parser.add_argument("--snapshot_path", type=str, default="./snapshots/")
    parser.add_argument("--log_path", type=str, default="./outputs/")

    """ HParams """
    # Main Training
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_decay_epochs",
        type=str,
        default="120,160",
    )
    # Linear Evaluation
    parser.add_argument("--eval_lr", type=float, default=0.01)
    parser.add_argument("--eval_n_epochs", type=int, default=30)

    # ER-AML hparams
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--buffer_neg", type=float, default=0)
    parser.add_argument("--incoming_neg", type=float, default=2.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.2)

    parser.add_argument("--projection_size", type=int, default=128)
    parser.add_argument("--projection_hidden_size", type=int, default=512)
    parser.add_argument("--prediction_hidden_size", type=int, default=256)

    # RePE hparams
    parser.add_argument("--prototypes_coef", type=float, default=1.0)
    parser.add_argument("--prototypes_lr", type=float, default=0.01)
    parser.add_argument("--distill_temp", type=float, default=2)

    # EWC
    parser.add_argument("--ewc_lambda", type=float, default=80)

    # ICARL & RePE hparams
    parser.add_argument("--distill_coef", type=float, default=1.0)

    # AsymTwins hparams
    parser.add_argument("--moving_average_decay", type=float, default=0.99)
    parser.add_argument("--asymtwins_method", type=str, default="simsiam")
    parser.add_argument("--ccp_alpha", type=float, default=0.9)

    # DER params
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)

    # MIR params
    parser.add_argument("--subsample", type=int, default=50)
    parser.add_argument("--mir_head_only", type=int, default=0)

    # CoPE params
    parser.add_argument("--cope_momentum", type=float, default=0.99)
    parser.add_argument("--cope_temperature", type=float, default=0.1)

    args = process_args(parser.parse_args())

    return args
