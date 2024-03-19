

from configs import *
import os
import sys
import numpy as np
import random
import torch

from torch.backends import cudnn
from torch import nn
sys.path.append('.')

from data import make_data_loader
from trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR,WarmupStepLR

from logger import setup_logger

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)




def train():
    
    # prepare dataset

    train_loader, val_loader, num_query, num_classes = make_data_loader()

    # prepare model
    model = build_model(num_classes)

    # if SOLVER_FINETUNE:
    #     model.load_state_dict(torch.load(TEST_WEIGHT).module.state_dict())
    # model = nn.DataParallel(model)


    optimizer = make_optimizer(model)
    scheduler = WarmupMultiStepLR(optimizer, SOLVER_STEPS, SOLVER_GAMMA, SOLVER_WARMUP_FACTOR,
                                  SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD)
    # scheduler = WarmupStepLR(optimizer,3, 9, SOLVER_WARMUP_FACTOR,
    #                               SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD)

    loss_func = make_loss()
    if MODEL_PRETRAIN_CHOICE == 'self':
        start_epoch = (eval(MODEL_PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])) / 241
        print('Start epoch:', start_epoch)
        path_to_optimizer = MODEL_PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        model.load_state_dict(torch.load(MODEL_PRETRAIN_PATH)["model"])
        model.to(MODEL_DEVICE)
        optimizer.load_state_dict(torch.load(path_to_optimizer)["optimizer"])
        # optimizer.to(MODEL_DEVICE)
        scheduler = WarmupMultiStepLR(optimizer, SOLVER_STEPS, 
                                        SOLVER_GAMMA, SOLVER_WARMUP_FACTOR,
                                        SOLVER_WARMUP_ITERS, SOLVER_WARMUP_METHOD, start_epoch)
    
    arguments = {}

    do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query
    )


def main():
    setup_seed(1)

    output_dir = OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("training")
    cudnn.benchmark = True
    train()


if __name__ == '__main__':
    main()
