from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

import torch
import models_vit

import numpy as np

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

class CustomDataset(Dataset):

    def __init__(self, anno_path, img_dir, transform=None):
        self.imgs = []
        self.labels = []
        with open(anno_path, 'r') as f:
            data = [line.replace('\n', '') for line in f.readlines()]
            for sample in data:
                img_name, label = sample.split(',')
                self.imgs.append(os.path.join(img_dir, img_name))
                self.labels.append(int(label))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
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
        return img, cropped, self.labels[index]

image_size = 224 
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_dataloader(anno_dir, img_dir, batch_size, transforms=val_transforms):
    dataset = CustomDataset(anno_dir, img_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return dataloader

def get_val_dataloader(anno_dir, img_dir, batch_size, transforms=val_transforms):
    dataset = CustomDataset(anno_dir, img_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader

def main():
    # model parameters
    model_name = 'vit_base_patch16'
    ckpt_model_1 = '../MAE-Face/checkpoints/epoch_14.pth'
    ckpt_model_2 = '../MAE-Face2/checkpoints/epoch_7.pth'
    global_pool = True # recommend: True for most cases, False if you want to evaluate the features from the pre-trained model without fine-tuning
    num_heads = 8 # specify the number of classification heads for the downstream task
    device = 'cuda'
    batch_size = 64
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
    
    batch_size = 144

    source_loader = get_train_dataloader('../download/train_set.txt', '../download/'
                                                  ,batch_size=batch_size)
    val_loader = get_val_dataloader('../download/val_set.txt', '../download/'
                                                 ,batch_size=batch_size)
    

    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        features = []
        labels = []
        n_iter = len(source_loader)
        for idx, (img, cropped, label) in enumerate(source_loader):
            img, cropped, label = img.to(device), cropped.to(device), label.to(device)

            _, feature_1 = model_1(img, ret_feature=True)
            _, feature_2 = model_2(cropped, ret_feature=True)

            feature = torch.cat((feature_1, feature_2), dim=1)


            features.extend(feature.cpu().numpy())
            labels.extend(label.cpu().numpy())

            print(f'Done [{idx}/{n_iter}]')
        
        features = np.array(features)
        labels = np.array(labels)
        np.save('features_train.npy', features)
        np.save('labels_train.npy', labels)

        #################################################################################
        features = []
        labels = []
        n_iter = len(source_loader)
        for idx, (img, cropped, label) in enumerate(val_loader):
            img, cropped, label = img.to(device), cropped.to(device), label.to(device)

            _, feature_1 = model_1(img, ret_feature=True)
            _, feature_2 = model_2(cropped, ret_feature=True)

            feature = torch.cat((feature_1, feature_2), dim=1)


            features.extend(feature.cpu().numpy())
            labels.extend(label.cpu().numpy())

            print(f'Done [{idx}/{n_iter}]')
        
        features = np.array(features)
        labels = np.array(labels)
        np.save('features_val.npy', features)
        np.save('labels_val.npy', labels)
        
        

         
if __name__=='__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
