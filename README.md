# Prototype-Sample Relation Distillation: Towards Replay-Free Continual Learning

You can find the pre-release code for [PRD](https://proceedings.mlr.press/v202/asadi23a/asadi23a.pdf) under `methods/repe.py`.

To reproduce the results in Figure 2 of the paper, you can run the following command:

Split-CIFAR100
```
python main.py --method=repe --dataset=cifar100 --data_root="path/to/data" --model=resnet18 --nf=64 --use_augs=1 --projection_size=128 --projection_hidden_size=512 --task_incremental=1 --keep_training_data=0 --multilinear_eval=0 --singlelinear_eval=1 --n_tasks=20 --n_epochs=100 --n_warmup_epochs=120 --batch_size=128 --lr=0.01 --supcon_temperature=0.1 --distill_coef=4.0 --prototypes_coef=2.0 --prototypes_lr=0.01 --distill_temp=1
```

Split-MiniImageNet
```
python main.py --method=repe --dataset=miniimagenet --data_root="path/to/data" --model=resnet18 --nf=64 --use_augs=1 --projection_size=128 --projection_hidden_size=512 --task_incremental=1 --keep_training_data=0 --multilinear_eval=0 --singlelinear_eval=1 --n_tasks=20 --n_epochs=100 --n_warmup_epochs=120 --batch_size=128 --lr=0.01 --supcon_temperature=0.1 --distill_coef=4.0 --prototypes_coef=2.0 --prototypes_lr=0.01 --distill_temp=1
```

---

# Probing Forgetting in Supervised and Unsupervised Continual Learning

This repository holds the experimental code for [Probing Representation Forgetting in Supervised and Unsupervised Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.pdf).

## (key) Requirements 
- Python 3.8
- Pytorch 1.9.0
- kornia 0.6.1
- fvcore 0.1.5
- wandb 0.12.7

## Running Experiments

```
python main.py --dataset <dataset> --method <method> --mem_size <mem_size> --eval_layer <model_stage> --keep_training_data <1 for probe eval> --model <model>  --n_tasks <num_tasks> --nf <model_width>
```

Example: Linear probe and CKA evaluation on finetuning(CE)

```
python main.py --dataset cifar100 --method er --mem_size 0 --eval_layer 4 --keep_training_data 1 --model resnet18  --n_tasks 10 --nf 32 --multilinear_eval 1 --cka_eval 1
```
Example: Linear probe evaluation on finetuning(SupCon)

```
python main.py --dataset cifar100 --method simclr --mem_size 0 --eval_layer 4 --keep_training_data 1 --model resnet18  --n_tasks 10 --nf 32 --multilinear_eval 1
```

Example: Observed accuracy on finetuning(CE)

```
python main.py --dataset cifar100 --method er --mem_size 0 --eval_layer 4 --keep_training_data 0 --model resnet18  --n_tasks 10 --nf 32 --singlelinear_eval 1
```


## Reproducing Results

SupCon training + LP evaluation

```
python main.py --batch_size=128  --buffer_batch_size=128 --dataset=cifar100 --download=1 --eval_lr=0.1 --eval_n_epochs=20 --task_incremental=1 --keep_training_data=1 --lr=0.05 --mem_size=0 --method=simclr --model=resnet18 --nf=32 --multilinear_eval=1 --n_epochs=100 --n_tasks=10 --save_snapshot=1 --singlelinear_eval=0 --data_root=data --snapshot_path=snapshots/  --use_augs=1
```


