import os
import sys
import json
import pickle
import numpy as np
import wandb

from datetime import datetime


class Logger:
    def __init__(self, args, save_every=30):

        self.step = 0
        self.args = args
        self.save_every = save_every

        if args.wandb:

            if args.wandb_mode == "offline":
                os.environ["WANDB_MODE"] = "offline"

            wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
            self.wandb = wandb
        else:
            self.wandb = None

        # create directory to store real logs
        date, time = datetime.now().strftime("%d_%m_%Y %H_%M_%S").split(" ")
        self.path = os.path.join(
            args.log_path, date, time + f"_{np.random.randint(1000)}"
        )
        os.makedirs(self.path)

        # dump args
        f = open(os.path.join(self.path, "params.json"), "w")
        json.dump(vars(args), f)
        f.close()

        self.to_pickle = {}
        self.picklename = os.path.join(self.path, "db.pickle")

    def register_name(self, name):
        if self.wandb is not None:
            self.wandb.config.update({"unique_name": name})

    def log_scalars(self, values):
        for k, v in values.items():
            self.to_pickle[str(self.step)] = values

        if self.wandb is not None:
            self.wandb.log(values, step=self.step)

        self.step += 1

    def log_line_series(self, n_tasks, accs):
        if self.wandb is not None:
            tasks = np.arange(n_tasks).tolist()
            keys = list(map(str, tasks))
            self.wandb.log(
                {
                    "custom_plot": self.wandb.plot.line_series(
                        xs=tasks,
                        ys=accs,
                        keys=keys,
                        title="Task Accuracies",
                        xname="task",
                    )
                }
            )

    def log_matrix(self, name, value):
        self.to_pickle[str(self.step)] += [(name, value)]
        self.step += 1

    def dump(self):
        f = open(self.picklename, "wb")
        pickle.dump(self.to_pickle, f)
        f.close()
        self.to_pickle = {}

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()

        self.dump()


if __name__ == "__main__":

    import wandb

    wandb.login(
        key="532994d10776b0e3449595d588b9e916cb6892ee"
    )  # TODO: Remove private key

    for dir_ in os.listdir("./outputs/"):
        for subdir in os.listdir("./outputs/" + dir_):
            print("./outputs/" + dir_ + subdir)
            f = open(
                f"./outputs/{dir_}/{subdir}/params.json",
            )
            params = json.load(f)
            f.close()

            f = open(f"./outputs/{dir_}/{subdir}/db.pickle", "rb")
            logs = pickle.loads(f.read())
            f.close()

            wandb.init(
                project=params["wandb_project"], name=params["exp_name"], config=params
            )
            for step in logs.keys():
                wandb.log(logs[step], step=int(step))
            wandb.finish()
