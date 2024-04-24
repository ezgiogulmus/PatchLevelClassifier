import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import os
import shutil
import json
import pandas as pd
import numpy as np
import random
from datetime import datetime
from fire import Fire
import wandb
from src import k_fold_split, Trainer

# data_folder = "/media/nfs/LN/"
data_folder = "/truba/home/fogulmus/"
def setup(seed, rank, world_size):
    # set seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def main(
        data_dir=data_folder+"CAMELYON16/images/",
        h5_dir=data_folder+"CAMELYON16/patches/",
        training_csv="./dataset_csv/train_data.csv",
        testing_csv="./dataset_csv/test_data.csv",
        run_name=None,
        backbone="resnet50",
        finetuning="mid",
        optimizer="Adam",
        lr=1e-3,
        l2_reg=1e-5,
        earlystopping=20,
        image_size=224,
        ext=".tif",
        patch_size=224,
        max_patches=100,
        augmentation=True,
        batch_size=128,
        val_batch_size=64,
        cross_validation=5,
        epochs=100,
        load_from=None,
        seed=7,
        multi_gpus=True,
        use_wandb=False
    ):
          
    config = locals()
    
    if not use_wandb:
        run_name = backbone+"_"+datetime.now().strftime("%m-%d_%H-%M") if run_name is None else run_name
        run_dir = os.path.join("./runs", run_name)
        os.makedirs(run_dir, exist_ok=True)
        config["run_dir"] = run_dir

        print("Training configuration:\n", config)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        shutil.copyfile("./main.py", os.path.join(run_dir, "main.py"))
        shutil.copyfile("./src.py", os.path.join(run_dir, "src.py"))
        world_size = torch.cuda.device_count() if multi_gpus else 1
        train_on_gpu(world_size, config, cross_validation, seed)
        
    else:
        os.environ["WANDB_API_KEY"] = "0c2a49295b00914e22f8cad553870dde0a95b577"
        parameter_dict = {
			'optimizer': {
				"values": ['Adam', 'SGD']
			},
			'backbone': {
				"values": ["resnet50", "densenet", "mobilenet", "vit", "ctp"]
			},
            'finetuning': {
				"values": [None, "deep", "mid", "shallow"]
			},
            'image_size': {
				"values": [224, 512]
			},
			'lr': {
				'distribution': 'uniform',
				'min': 2e-4,
				'max': 1e-1
			},
			'l2_reg': {
				'distribution': 'uniform',
				'min': 1e-5,
				'max': 1e-3
			}
		}
        initial_config = {k: {"value": v} for k, v in config.items() if k not in parameter_dict.keys()}
        parameter_dict.update(initial_config)
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'test_acc',
                'goal': 'maximize'
            },
            'parameters': parameter_dict
        }
        sweep_id = wandb.sweep(sweep_config, project=run_name) 
        wandb.agent(sweep_id, function=train_on_gpu)

def train_on_gpu(world_size=None, config=None, cross_validation=5, seed=7):
    world_size = torch.cuda.device_count() if world_size is None else world_size
    if world_size == 1:
        run_training(0, 1, config, cross_validation, seed)
    else:
        print("Training on {} GPU." .format(world_size))
        mp.spawn(run_training, nprocs=world_size, args=(world_size, config, cross_validation, seed))

def run_training(rank=0, world_size=1, config=None, cross_validation=5, seed=7):
    is_main = rank == 0
    is_ddp = world_size > 1
    if config is None:
        wandb.init()
        config = wandb.config
        run_name = wandb.run.name
        run_dir = os.path.join("./runs/wandb/", run_name)
        os.makedirs(run_dir, exist_ok=True)
        config["run_dir"] = run_dir

    config.update(dict(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    ))

    df = pd.read_csv(config["training_csv"])
    k_fold = 5 if cross_validation < 0 else cross_validation
    splits = k_fold_split(df, k_fold)
    
    label_dict = {name: x for x, name in enumerate(sorted(pd.unique(df["label"])))}
    if is_main:
        print("Classification labels:\n", label_dict)
    
    cv_results = None
    for cv in range(len(splits)):
        if is_main:
            print(f"Starting CV {cv+1}")
        if is_ddp:
            setup(seed, rank, world_size)
            print(f"{rank + 1}/{world_size} process initialized.")
        
        train_df, val_df = splits[cv]["train"].reset_index(drop=True), splits[cv]["val"].reset_index(drop=True)
        train_ratios = dict(train_df["label"].value_counts()/train_df["label"].value_counts().sum())
        val_ratios = dict(val_df["label"].value_counts()/val_df["label"].value_counts().sum())
        if is_main:
            print("Train distribution on slide level:")
            [print("\t", k, round(v, 2)) for k, v in sorted(train_ratios.items())]
            print("\nValidation distribution on slide level:")
            [print("\t", k, round(v, 2)) for k, v in sorted(val_ratios.items())]
        trainer = Trainer(**config, train_df=train_df, val_df=val_df, label_dict=label_dict, cv=cv)
        trainer.run()
        results = trainer.evaluate()
        if config["use_wandb"]:
            wandb.log(results)
            break
        if cv_results is None:
            cv_results = {k: [] for k in results.keys()}
        for k, v in results.items():
            cv_results[k].append(round(v, 4))

        pd.DataFrame(cv_results).to_csv(os.path.join(trainer.run_dir, "cv_results.csv"))
        torch.cuda.empty_cache()
        if is_ddp:
            dist.destroy_process_group()
        if cross_validation < 0:
            break

if __name__ == "__main__":
    Fire(main)