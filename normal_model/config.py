# config.py

import argparse
import torch
import random
import numpy as np

def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args(auto=True):
    args = argparse.Namespace()
    args.n_classes = 8

    args.model_name = 'vit_base_patch16'
    args.ckpt_path = './mae_face_pretrain_vit_base.pth'
    args.global_pool = True

    args.train_batch_size = 288
    args.val_batch_size = 144
    args.image_size = 224
    args.seed = 1

    args.learning_rate = 1e-4
    args.epochs = 20
    args.eval = False
    args.ckpt = "logs/model"
    args.save_ckpt = "checkpoints/"

    return args