import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

LOW_IMG_WIDTH = 64
LOW_IMG_HEIGHT = 64


class ImageDataset(Dataset):
    def __init__(self, img_dir, is_train=True):
        self.resize = transforms.Compose(
            transforms.Resize((LOW_IMG_WIDTH, LOW_IMG_HEIGHT),
                              antialias=True)
        )
        self.is_train = is_train
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def normalize(self, input_image, target_image):
        input_image = (input_image - 0.5) / 0.5
        target_image = (target_image - 0.5) / 0.5
        return input_image, target_image

    def augment(self, input_image, target_image):
        """
        Perform augmetation techniques (horizontal or vertical flipping, rotation)
        """
        flip = transforms.Compose(
            transforms.RandomHorizontalFlip(0.4),
            transforms.RandomVerticalFlip(0.3)
        )

        rotate = transforms.RandomRotation(30)

        input_image = flip(input_image)
        target_image = flip(target_image)

        if torch.rand([]) < 0.5:
            input_image = rotate(input_image)
            target_image = rotate(target_image)

        return input_image, target_image

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert('RGB'))
        image = transforms.ToTensor(image)
        input_image = self.resize(image).type(torch.float32)
        target_image = image.type(torch.float32)

        input_image, target_image = self.normalize(input_image, target_image)

        if self.is_train:
            input_image, target_image = self.augment(input_image, target_image)

        return input_image, target_image
