import argparse
import os

import json5
import numpy as np
import torch

from torch.utils.data import DataLoader
from util.utils import initialize_config


def main(config, resume):
    torch.manual_seed(config["seed"])  # torch和numpy都要设置随机数种子
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],   # batch_size
        num_workers=config["train_dataloader"]["num_workers"],   # 子进程数
        shuffle=config["train_dataloader"]["shuffle"],     # 是否shuffle
        pin_memory=config["train_dataloader"]["pin_memory"]  # 是否加载数据样本到cuda固定内存中
    )

    valid_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        num_workers=1,
        batch_size=1
    )

    model = initialize_config(config["model"])

    # 如果改optim在config中
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])
    # 如果改loss_function改json中的loss_function中的main和model.loss

    trainer_class = initialize_config(config["trainer"], pass_args=False)
    # 只传了trainer.py中的Trainer类，没传参数，下面传参

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wave-U-Net for Speech Enhancement")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    # 实验名砍掉base根目录以及扩展名
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)
