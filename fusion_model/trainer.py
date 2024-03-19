# trainer.py

import torch
import torch.nn as nn

import torchvision.transforms as transforms

import os
from loader import get_train_dataloader, get_val_dataloader
from sklearn.metrics import f1_score, accuracy_score

from model import Fusion


class Trainer:
    def __init__(self, args, device, use_amp=True):
        self.args = args
        self.device = device
        self.counterk=0
        self.n_classes = args.n_classes

        # Caffe Alexnet for singleDG task, Leave-one-out PACS DG task.
        # self.extractor = caffenet(args.n_classes).to(device)
        self.model = Fusion(self.n_classes).to(device)

        self.source_loader = get_train_dataloader('./data/features_train.npy', './data/labels_train.npy'
                                                  ,batch_size=args.train_batch_size)
        self.val_loader = get_val_dataloader('./data/features_val.npy', './data/labels_val.npy'
                                                 ,batch_size=args.val_batch_size)

        self.len_dataloader = len(self.source_loader)

        # Get optimizers and Schedulers, self.discriminator
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.args.epochs * 1))


        # Set up Automatic Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp


    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        print(f'[{epoch + 1}/{self.args.epochs}]')
        for idx, (data, label) in enumerate(self.source_loader):
            data, label = data.to(self.device), label.to(self.device)
            feature_1 = data[:, :768]
            feature_2 = data[:, 768:]
            # Stage 1
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):

                self.optimizer.zero_grad()

                outputs = self.model(feature_1, feature_2)

                # Total loss & backward
                loss = criterion(outputs, label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            print(f'Iter [{idx}/{self.len_dataloader}] Loss: {loss}')

            del loss

        self.model.eval()
        with torch.no_grad():
            acc, f1 = self.do_test(self.val_loader)
            self.results["val"][self.current_epoch] = acc
            print(f'Val Acc: {acc}')
            with open(os.path.join(self.args.save_ckpt, 'log.txt'), 'a+') as file:
                file.write(f'[Epoch {epoch + 1}] Accuracy: {acc} - F1-Score: {f1}\n')

    def do_test(self, loader):
        n_inter = len(loader)
        all_labels = []
        all_preds = []
        for idx, (data, label) in enumerate(loader):
            data, label = data.to(self.device), label.to(self.device)
            feature_1 = data[:, :768]
            feature_2 = data[:, 768:]
            outputs = self.model(feature_1, feature_2)

            _, preds = outputs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            print(f'DONE [{idx + 1}/{n_inter}]')

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Test Accuracy: {accuracy}, Macro F1 Score: {f1}')

        return accuracy, f1

    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        for self.current_epoch in range(self.args.epochs):
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
            if self.results["val"][self.current_epoch] > current_high:
                print('Saving Best model ...')
                torch.save(self.model.state_dict(), os.path.join(self.args.save_ckpt, 'best_model.pth'))
                current_high = self.results["val"][self.current_epoch]
            if (self.current_epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.args.save_ckpt, f'epoch_{self.current_epoch + 1}.pth'))

        val_res = self.results["val"]
        print("Best val %g" % (val_res.max()))