import os
import json

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class COCODataset(Dataset):

    def __init__(self, image_dir, label_file):

        self.image_dir = image_dir

        with open(label_file, "r") as f:
            self.image_labels = json.load(f)

        self.image_names = list(self.image_labels.keys())

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        label = self.image_labels[image_name]

        return image, label