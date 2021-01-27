'''
PyTorch Dataset Handling. The dataset folder should comprise of two subfolders namely "train" and "test" where both folders has subfolders that named
according to their class names.
'''

import os
import glob
import cv2
import torch
from torch.utils import data
from torch.utils.data import Dataset, dataset


class LoadDataset(Dataset):
    '''Loads the dataset from the given path.
    '''

    def __init__(self, dataset_folder_path, image_size=224, image_depth=3, train=True, transform=None):
        '''Parameter Init.
        '''

        assert not dataset_folder_path is None, "Path to the dataset folder must be provided!"

        self.dataset_folder_path = dataset_folder_path
        self.transform = transform
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()


    def get_classnames(self):
        '''Returns the name of the classes in the dataset.
        '''
        return os.listdir(f"{self.dataset_folder_path.rstrip('/')}/train/" )


    def read_folder(self):
        '''Reads the folder for the images with their corresponding label (foldername).
        '''

        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/train/"
        else:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/test/"

        for x in glob.glob(folder_path + "**", recursive=True):

            if not x.endswith('jpg'):
                continue

            class_idx = self.classes.index(x.split('/')[-2])
            image_path_label.append((x, int(class_idx)))

        return image_path_label


    def __len__(self):
        '''Returns the total size of the data.
        '''
        return len(self.image_path_label)

    def __getitem__(self, idx):
        '''Returns a single image and its corresponding label.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.image_path_label[idx]

        if self.image_depth == 1:
            image = cv2.imread(image, 0)
        else:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }


class LoadInputImages(Dataset):
    '''Loads the dataset for visualization.
    '''

    def __init__(self, input_folder, image_size, image_depth, transform=None):
        '''Param init.
        '''
        self.input_folder = input_folder.rstrip('/') + '/'
        self.image_size = image_size
        self.image_depth = image_depth
        self.transform = transform

        self.image_paths = self.read_folder()



    def read_folder(self):
        '''Reads all the image paths in the given folder.
        '''
        image_paths = []
        for x in glob.glob(self.input_folder + '**'):

            if not x.endswith('jpg'):
                continue
            image_paths.append(x)

        return image_paths


    def __len__(self):
        '''Returns the total number of images in the folder.
        '''
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''Returns a single image array.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image_paths[idx]

        if self.image_depth == 1:
            image = cv2.imread(image, 0)
        else:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return image
