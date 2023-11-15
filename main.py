import time
import numpy as np
from collections import OrderedDict as OD

from data.base import *
from models import SupConResNet, SupCEResNet, MultiHeadResNet
from methods import *
from utils import Logger, get_parser, set_seed
from utils import adjust_learning_rate, load_best_args


def main():
    # >>>>> Arguments <<<<<
    args = get_parser()

    if args.seed is not None:
        set_seed(args.seed)

    if args.load_best_args:
        load_best_args(args)

    if args.method in ["iid", "iid++"]:
        print("Overwriting args for iid setup")
        args.n_tasks = 1
        args.mem_size = 0

    # >>>>> Obligatory overhead <<<<<
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda_id}"
        print(
            f"\nNumber of available GPUs : {torch.cuda.device_count()}\nUsing: {device}\n"
        )
    else:
        device = "cpu"

    # make dataloaders
    train_tf, train_loader, val_loader, test_loader = get_data_and_tfs(args)
    train_tf.to(device)

    logger = Logger(args=args)
    args.mem_size = args.mem_size * args.n_classes

    # for iid methods
    args.train_loader = train_loader

    eval_accs = []
    if args.validation:
        mode = "valid"
        eval_loader = val_loader
    else:
        mode = "test"
        eval_loader = test_loader

    if args.keep_training_data:
        loader = (train_loader, eval_loader)
    else:
        loader = eval_loader

    # >>>>> Classifier  <<<<<
    if args.method in SSL_METHODS or args.method in ["repe", "supcontta", "spc"]:
        model = SupConResNet(
            name=args.model,
            head="mlp",
            nf=args.nf,
            input_size=args.input_size,
            feat_dim=args.projection_size,
            hidden_dim=args.projection_hidden_size,
            batch_norm=True
            if (args.method in ["simsiam", "barlowtwins", "vicreg"])
            else False,
            num_layers=3,
        )
    elif args.task_incremental:
        model = MultiHeadResNet(
            name=args.model,
            nf=args.nf,
            input_size=args.input_size,
            n_classes_per_head=args.n_classes_per_task,
            n_heads=args.n_tasks,
        )
    else:
        model = SupCEResNet(
            name=args.model,
            head="distlinear" if (args.method in ["er_ace", "er_aml"]) else "linear",
            nf=args.nf,
            input_size=args.input_size,
            num_classes=args.n_classes,
        )

    model = model.to(device)
    model.train()

    agent = get_method(args.method)(model, logger, train_tf, args)
    n_params = sum(np.prod(p.size()) for p in model.parameters())
    print("Number of classifier parameters:", n_params)

    # >>> Train/Evaluate  Model <<<
    if not args.use_snapshots:
        eval_accs = train(
            args,
            agent=agent,
            train_loader=train_loader,
            eval_loader=loader,
            device=device,
            mode=mode,
        )
    else:
        eval_accs = evaluate_snapshots(args, agent=agent, loader=loader, mode=mode)

    # >>> Log Final Results <<<
    log(logger, eval_accs, mode)

    logger.close()


def train(args, agent, train_loader, eval_loader, device, mode):
    """ """
    eval_accs = []

    start_task = 0
    if args.half_iid:
        start_task = (args.n_tasks // 2) - 1

    for task in range(start_task, args.n_tasks):
        # set task
        train_loader.sampler.set_task(task, sample_all_seen_tasks=(task == start_task))

        agent.train()
        agent.on_task_start()

        n_seen = 0
        start = time.time()

        n_epochs = args.n_epochs
        if task == start_task:
            n_epochs += args.n_warmup_epochs

        print("\n>>> Task #{} --> Model Training".format(task))
        for epoch in range(n_epochs):

            adjust_learning_rate(args, agent.opt, epoch)

            for i, (x, y) in enumerate(train_loader):
                if n_seen > args.samples_per_task > 0:
                    break

                if "cuda" in device:
                    cuda_device = torch.device(device)
                    x = x.to(cuda_device, non_blocking=True)
                    y = y.to(cuda_device, non_blocking=True)

                inc_data = {"x": x, "y": y, "t": task}
                loss = agent.observe(inc_data)

                n_seen += x.size(0)

                print(
                    f"Epoch: {epoch + 1} / {n_epochs} | {i} / {len(train_loader)} - Loss: {loss}",
                    end="\r",
                )

            if (epoch + 1) % args.eval_every == 0 or (epoch + 1 == n_epochs):
                print(f"Task {task}. Time {time.time() - start:.2f}")
                eval_accs += [agent.eval_agent(eval_loader, task, mode=mode)]
                agent.train()

        agent.on_task_finish(task)

    return eval_accs


def evaluate_snapshots(args, agent, loader, mode):
    # Load pretrained agent from local file
    agent.load_model_history()

    eval_accs = []
    for task in range(args.n_tasks):
        agent.model = agent.model_history[str(task)]

        eval_accs += [agent.eval_agent(loader, task, mode=mode)]

    return eval_accs


def log(logger, eval_accs, mode):
    # ----- Final Results ----- #

    accs = np.stack(eval_accs).T
    avg_acc = accs[:, -1].mean()
    avg_fgt = (accs.max(1) - accs[:, -1])[:-1].mean()

    print("\nFinal Results\n")
    print(f"Avg Acc: {avg_acc} - Avg Fgt: {avg_fgt}")
    # logger.log_matrix(f'{mode}_acc', accs)
    logger.log_scalars(
        {
            f"{mode}/avg_acc": avg_acc,
            f"{mode}/avg_fgt": avg_fgt,
            # 'train/n_samples': n_seen,
            # 'metrics/model_n_bits': n_params * 32,
            # 'metrics/cost': agent.cost,
            # 'metrics/one_sample_flop': agent.one_sample_flop,
            # 'metrics/buffer_n_bits': agent.buffer.n_bits()
        }
    )


if __name__ == "__main__":
    main()
