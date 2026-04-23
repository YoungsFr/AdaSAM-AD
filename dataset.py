import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import random
import numpy as np

from PIL import Image

class PolypSegmentDataset(Dataset):
    def __init__(self, data_path, img_size=336, train=True):
        super().__init__()
        self.train = train
        assert os.path.exists(data_path)
        self.data_path = data_path
        self.img_size = img_size
            
        self.data = self.get_training_data() if train is True else self.get_testing_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, mask_path = self.data[index]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(0.5, 1.3), ratio=(0.75, 1.33))
            image = F.resized_crop(image, i, j, h, w, (self.img_size, self.img_size))
            mask = F.resized_crop(mask, i, j, h, w, (self.img_size, self.img_size), InterpolationMode.NEAREST)
            # image = F.resize(image, (self.img_size, self.img_size), interpolation=F.InterpolationMode.BILINEAR)
            # mask = F.resize(mask, (self.img_size, self.img_size), interpolation=F.InterpolationMode.NEAREST)
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
                
            angle = random.uniform(-15, 15)
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            
            if random.random() > 0.5:
                gamma = random.uniform(0.8, 1.2)
                image = F.adjust_gamma(image, gamma)
            
            if random.random() > 0.2:
                brightness = random.uniform(0.7, 1.3)
                contrast = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                hue = random.uniform(-0.05, 0.05) 
                
                image = F.adjust_brightness(image, brightness)
                image = F.adjust_contrast(image, contrast)
                image = F.adjust_saturation(image, saturation)
                image = F.adjust_hue(image, hue)

            if random.random() > 0.3:
                sigma = random.uniform(0.1, 2.0)
                image = F.gaussian_blur(image, kernel_size=(5, 5), sigma=sigma)
        else:
            image = F.resize(image, (self.img_size, self.img_size), interpolation=F.InterpolationMode.BILINEAR)
            mask = F.resize(mask, (self.img_size, self.img_size), interpolation=F.InterpolationMode.NEAREST)

        image = F.to_tensor(image)
        image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        mask = F.to_tensor(mask)

        return image, mask, img_path

    def get_training_data(self):
        print('start processing training data')
        img_path = os.path.join(self.data_path, 'image')
        mask_path = os.path.join(self.data_path, 'mask')
        print(img_path)
        assert os.path.exists(img_path)
        assert os.path.exists(mask_path)

        img_files = sorted(os.listdir(img_path))
        mask_files = sorted(os.listdir(mask_path))
        data = []
        
        for img, mask in zip(img_files, mask_files):
            data.append((
                os.path.join(img_path, img),
                os.path.join(mask_path, mask),
            ))
        return data
    
    def get_testing_data(self):
        
        path = self.data_path
        print(f'Start processing testing data from: {path}')
        
        img_path = os.path.join(path, 'image')
        mask_path = os.path.join(path, 'mask')
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Test image path not found: {img_path}")

        img_files = sorted(os.listdir(img_path))
        mask_files = sorted(os.listdir(mask_path))
        data = []

        for img, mask in zip(img_files, mask_files):
            data.append((
                os.path.join(img_path, img),
                os.path.join(mask_path, mask),
            ))
        return data

def get_train_val_loaders(data_path, val_path, img_size=512,batch_size=32, num_workers=1):
    train_set = PolypSegmentDataset(data_path, img_size, train=True)
    val_set = PolypSegmentDataset(val_path, img_size, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


def get_test_loader(data_path, img_size=512, batch_size=1, num_workers=4):

    test_dataset = PolypSegmentDataset(
        data_path=data_path,
        img_size=img_size,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f'Successfully loaded {len(test_dataset)} test samples.')
    return test_loader