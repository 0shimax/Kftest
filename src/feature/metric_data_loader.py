from pathlib import Path
import random
import numpy
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import multiprocessing

from feature.utils import ImageTransform, GcsIO


class WBCDataset(Dataset):
    def __init__(self, n_class, image_labels, root_dir,
                 subset="Dataset1", transform=ImageTransform(),
                 project="<your project id>", bucket_name="kf-test1234",
                 train=True):
        super().__init__()
        self.image_labels = image_labels
        self.root_dir = root_dir
        self.subset = subset
        self.gcs_io = GcsIO(project, bucket_name)
        self.transform = transform

        self.n_class = n_class
        self.train = train
        self.n_relation = self.n_class**2

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]
        image = self._read_image(img_name)
        image = self.transform(image)

        if self.train:
            near_image = self._get_near_image(label)
            far_image, far_label = self._get_far_image_and_label(label)

            near_relational_tag = label*(self.n_class-1) + label
            far_relational_tag = label*(self.n_class-1) + far_label

            label = torch.LongTensor([label])
            far_label = torch.LongTensor([far_label])
            near_relational_tag = torch.LongTensor([near_relational_tag])
            far_relational_tag = torch.LongTensor([far_relational_tag])
            return image, label, near_image, label, far_image, far_label, near_relational_tag, far_relational_tag
        else:
            return image, label

    def _get_far_image_and_label(self, near_category):
        idxs = numpy.where(self.image_labels[:,1]!=near_category)[0]
        random.shuffle(idxs)
        idx = idxs[0]
        img_name, label = self.image_labels[idx]
        image = self._read_image(img_name)
        image = self.transform(image)
        return image, label

    def _get_near_image(self, near_category):
        idxs = numpy.where(self.image_labels[:,1]==near_category)[0]
        random.shuffle(idxs)
        idx = idxs[0]

        img_name = self.image_labels[idx][0]
        image = self._read_image(img_name)
        return self.transform(image)

    def _read_image(self, img_name):
        image_path = '/'.join([self.root_dir, self.subset, "{0:03}.bmp".format(img_name)])
        image = self.gcs_io.load_image(image_path)
        return image


def loader(dataset, batch_size,  shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=multiprocessing.cpu_count())
    return loader