from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
    
class NormalImage(Dataset):

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
        img_path = os.path.join(self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, self.labels[index]


image_size = 224
# image_size = 288 #convnextv2
# image_size = 600 #b7

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((image_size, image_size), (0.8, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_dataloader(anno_dir, img_dir, batch_size, transforms=train_transforms):
    dataset = NormalImage(anno_dir, img_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return dataloader

def get_val_dataloader(anno_dir, img_dir, batch_size, transforms=val_transforms):
    dataset = NormalImage(anno_dir, img_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader