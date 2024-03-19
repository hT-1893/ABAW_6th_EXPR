from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

   
class CroppedImage(Dataset):

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

        img = get_concat_v(eye, mouth)
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
    dataset = CroppedImage(anno_dir, img_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return dataloader

def get_val_dataloader(anno_dir, img_dir, batch_size, transforms=val_transforms):
    dataset = CroppedImage(anno_dir, img_dir, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader