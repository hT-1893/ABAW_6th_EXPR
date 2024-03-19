# run.py

import torch
import os

from trainer import Trainer
from config import get_args, fix_all_seed

def main():
    args = get_args()
    fix_all_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    if not args.eval:
        trainer.do_training()
    else:
        trainer.extractor.load_state_dict(torch.load(''))
        trainer.do_eval()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()