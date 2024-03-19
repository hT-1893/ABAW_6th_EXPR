from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

import torch
import models_vit

import numpy as np

from Fusion.model import Fusion

from collections import Counter

from sklearn.metrics import f1_score, accuracy_score

def count(x1, x2, x3):
    results = []
    for i in range(len(x1)):
        counter = Counter([x1[i], x2[i], x3[i], x3[i]])
        most_common_value = counter.most_common(1)[0][0]
        results.append(most_common_value)
    return torch.tensor(results)
    

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

class CustomDataset(Dataset):

    def __init__(self, anno_path, img_dir, transform=None):
        self.imgs = []
        self.img_dir = img_dir
        with open(anno_path, 'r') as f:
            data = [line.replace('\n', '') for line in f.readlines()[1:]]
            for sample in data:
                img_name, _ = sample.split(',')
                if os.path.exists(os.path.join(self.img_dir, img_name)):
                    self.imgs.append(img_name)

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = os.path.join(self.img_dir, self.imgs[index])
        img = Image.open(path).convert('RGB')
        w, h = img.size
        eye = img.crop((int(w * 0.2), int(h * 0.35), int(w * 0.8), int(h * 0.55)))

        img = Image.open(path).convert('RGB')
        mouth = img.crop((int(w * 0.2), int(h * 0.7), int(w * 0.8), int(h * 0.9)))

        cropped = get_concat_v(eye, mouth)
        img = Image.open(path).convert('RGB')

        if self.transform != None:
            img = self.transform(img)
            cropped = self.transform(cropped)
        return img, cropped

image_size = 224 
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_val_dataloader(anno_dir, img_dir, batch_size, transforms=val_transforms):
    dataset = CustomDataset(anno_dir, img_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader, dataset

def main():
    # model parameters
    model_name = 'vit_base_patch16'
    ckpt_model_1 = './mae_checkpoint/normal/epoch_14.pth'
    ckpt_model_2 = './mae_checkpoint/cropped/epoch_7.pth'
    ckpt_model_fusion = './Fusion/checkpoints/best_model.pth'
    global_pool = True # recommend: True for most cases, False if you want to evaluate the features from the pre-trained model without fine-tuning
    num_heads = 8 # specify the number of classification heads for the downstream task
    device = 'cuda'
    # create model
    model_1 = getattr(models_vit, model_name)(
        global_pool=global_pool,
        num_classes=num_heads,
        drop_path_rate=0.1,
        img_size=224,
    )
    model_2 = getattr(models_vit, model_name)(
        global_pool=global_pool,
        num_classes=num_heads,
        drop_path_rate=0.1,
        img_size=224,
    )

    fusion_model = Fusion(num_heads)

    print(f"Load pre-trained checkpoint from: {ckpt_model_1}")
    checkpoint = torch.load(ckpt_model_1, map_location='cpu')
    msg = model_1.load_state_dict(checkpoint, strict=False)
    print(msg) # print which weights are not loaded
    model_1.to(device)

    print(f"Load pre-trained checkpoint from: {ckpt_model_2}")
    checkpoint = torch.load(ckpt_model_2, map_location='cpu')
    msg = model_2.load_state_dict(checkpoint, strict=False)
    print(msg) # print which weights are not loaded
    model_2.to(device)

    print(f"Load pre-trained checkpoint from: {ckpt_model_fusion}")
    checkpoint = torch.load(ckpt_model_fusion, map_location='cpu')
    msg = fusion_model.load_state_dict(checkpoint, strict=False)
    print(msg) # print which weights are not loaded
    fusion_model.to(device)


    # DATA LOADER
    batch_size = 512
    FILE_FORMAT_PATH = '../download/example_submission/predictions.txt'      # Path to the format .txt
    IMG_PATH = '../download/frames/'             # Path to folder image: EX: '../download/frames/100-29-1080x1920/00001.jpg'
    val_loader, dataset = get_val_dataloader(FILE_FORMAT_PATH, IMG_PATH
                                                 ,batch_size=batch_size)
    

    model_1.eval()
    model_2.eval()
    fusion_model.eval()
    preds = []
    with torch.no_grad():
        n_iter = len(val_loader)
        for idx, (img, cropped) in enumerate(val_loader):
            img, cropped = img.to(device), cropped.to(device)

            output_1, feature_1 = model_1(img, ret_feature=True)
            # _, pred_1 = output_1.max(dim=1)

            output_2, feature_2 = model_2(cropped, ret_feature=True)
            # _, pred_2 = output_2.max(dim=1)

            output_3 = fusion_model(feature_1, feature_2)
            _, pred_3 = output_3.max(dim=1)

            # pred = count(pred_1, pred_2, pred_3)
            pred = pred_3

            preds.extend(pred.cpu().numpy())

            print(f'Done [{idx}/{n_iter}]')

    frame_ids = dataset.imgs
    results = []
    for idx in range(len(frame_ids)):
        results.append([frame_ids[idx], preds[idx]])

    with open('prediction.txt', 'w') as f:
        for result in results:
            line = f'{result[0]},{int(result[1])}\n'
            f.write(line)
        
        

         
if __name__=='__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
