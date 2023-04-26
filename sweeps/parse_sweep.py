from typing import Any, List
import argparse
import itertools
import numpy as np
import os
import yaml


def get_sbatch_command(
    account: str = "rrg-eugenium",
    job_name: str = "SCRD",
    gres: int = 1,
    cpus_per_task: int = 8,
    mem: int = 32,
    time: str = "23:59:00",
):
    return f"""#!/bin/bash

#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --gres=gpu:{gres}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}G
#SBATCH --time={time}
#SBATCH -o /home/nasadi/scratch/slurm-%j.out

# 1. Create your environement locally
module load python/3.8.2
source $HOME/env/bin/activate

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
tar -zxf $HOME/projects/rrg-eugenium/nasadi/cl-datasets.tar.gz -C $SLURM_TMPDIR
echo "Copied dataset successfully!"

# WandB setup 
wandb offline

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
cd $HOME/codes/lifelong-ssl/

"""


def main(
    sweep_path: str,
    target_path: str,
    bash_file_name: str,
    n_bash_files: int,
    account: str,
    job_name: str,
    gres: int,
    cpus_per_task: int,
    mem: int,
    time: str,
):
    with open(sweep_path, "r") as f:
        config = yaml.safe_load(f)

    base_command = "python main.py"
    variable_args = []

    for key, param_value in config["parameters"].items():

        if "value" in param_value.keys():
            base_command += f" --{key}={param_value['value']}"
        else:
            variable_args.append([f" --{key}={item}" for item in param_value["values"]])

    variable_args = list(itertools.product(*variable_args))
    print(
        f"Creating {len(variable_args)} run commands from `{sweep_path}`\n\nOutput bash files:"
    )

    final_commands = [(base_command + "".join(args) + "\n") for args in variable_args]
    final_commands = np.array_split(final_commands, n_bash_files)

    for idx, batch_commands in enumerate(final_commands):

        sbatch_command = get_sbatch_command(
            account, job_name, gres, cpus_per_task, mem, time
        ) + "".join(batch_commands)

        base_path = target_path + f"/{bash_file_name}/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        file_path = base_path + f"{bash_file_name}_{idx + 1}.sh"
        with open(file_path, "w") as f:
            f.writelines(sbatch_command)
        print(f"`{file_path}` with {len(batch_commands)} runs")

    with open(f"{target_path}/sbatch_{bash_file_name}.sh", "w") as f:
        f.writelines(
            [
                f"sbatch ./{bash_file_name}/{bash_file_name}_{i + 1}.sh\n"
                for i in range(idx + 1)
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_path", type=str)
    parser.add_argument("--target_path", type=str, default="./sweeps/scripts")
    parser.add_argument("--bash_file_name", type=str, default="bash")
    parser.add_argument("--n_bash_files", type=int, default=1)

    parser.add_argument("--account", type=str, default="rrg-eugenium")
    parser.add_argument("--job_name", type=str, default="SCRD")
    parser.add_argument("--gres", type=int, default=1)
    parser.add_argument("--cpus_per_task", type=int, default=8)
    parser.add_argument("--mem", type=int, default=32)
    parser.add_argument("--time", type=str, default="23:59:00")

    args = parser.parse_args()

    main(
        sweep_path=args.sweep_path,
        target_path=args.target_path,
        bash_file_name=args.bash_file_name,
        n_bash_files=args.n_bash_files,
        account=args.account,
        job_name=args.job_name,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
    )
