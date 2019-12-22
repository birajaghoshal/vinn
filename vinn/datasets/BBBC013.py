import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import os
import h5py
import torchvision.transforms.functional as F

class AddDummyLayer(object):
    """Adds a dummy layer to the input image."""   
    def __call__(self, image):
        """
        image must use numpy image dimension encoding: (width, height, n_channels)
        """
        return np.dstack((image, np.zeros(image.shape[:2]))).astype(np.uint8)
    
class RemoveDummyLayer(object):
    """Removes the dummy layer from the input image."""
    def __call__(self, image):
        """
        image must use tensor image dimension encoding: (n_channels, width, height)
        """
        return image[:2,:,:]
    
class Random90Flip(object):
    """Flip by 90 degrees the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if np.random.rand() < self.p:
            return F.rotate(img, 90)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class BBBC013(Dataset):
    """BBBC013 dataset <https://data.broadinstitute.org/bbbc/BBBC013/>"""
    
    csv_file = "/da/isld/share/deeplearning/datasets/public/BBBC013/metadata.csv"
    img_file = "/da/isld/share/deeplearning/datasets/public/BBBC013/images.npy"
    
    n_augmentations = 100
    img_size = 256
    aug_transform = transforms.Compose([AddDummyLayer(),
                                        transforms.ToPILImage(),
                                        transforms.RandomRotation(degrees=20.0),
                                        transforms.RandomCrop(size=img_size, padding=None, pad_if_needed=True),
                                        Random90Flip(p=0.5),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        RemoveDummyLayer()])
    
    group_names = ['negative', 'positive', 'wortmannin', 'LY294.002']
    compound_names = ['NA', 'wortmannin', 'wortmannin', 'LY294.002']
    
    def __init__(self, train=True, binary=True, download=False, transform=None, target_transform=None, data_path="data/BBBC013/BBBC013_aug500.hdf5"):
        """
        Args:
            train (bool, optional): If True, selects the training data (75% of the total),
                otherwise it selects the remaining part.
            binary (bool, optional): If True, selects only the 'negative' and 'positive'
                groups, otherwise all the groups are selected.
            download (bool, optional): If true, downloads the dataset from the Isilon to
                data_path using csv_file and img_file for the metadata and the images 
                respectively.
                The dataset is augmented before being downloaded, look at download() for
                more information.
                If dataset is already downloaded, it is not downloaded again.
            transform (callable, optional): A function/transform to be applied on a sample
                numpy image with format [C, W, H].
            target_transform (callable, optional): A function/transform to be applied on a
                sample target with the format (group, concentration).
            data_path (string, optional): Path to the dataset hdf5 file.
        """ 
    
        if download:
            self.download(data_path)

        if not os.path.exists(data_path):
            raise RuntimeError("Dataset not found at '{}'. Use download=True to download it.".format(data_path))
    
        self.binary = binary
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.data_path = data_path
        self.dataset_file = None
        self.n_items = None
        
        with h5py.File(self.data_path, 'r') as dataset_file:
            if self.train:
                dataset = dataset_file['BBBC013/train']
            else:
                dataset = dataset_file['BBBC013/test']

            if self.binary:
                self.n_items = dataset["binary_index_table"].shape[0]
            else:
                self.n_items = dataset["images"].shape[0]
        
    def __len__(self):
        return self.n_items

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, (group, concentration)) where group is an index that can be used to
                        obtain information from the group_names and compound_names lists.
        """
        if self.binary:
            index = self.dataset["binary_index_table"][index]
        
        image = self.dataset["images"][index]
        group = self.dataset["groups"][index]
        concentration = self.dataset["concentrations"][index]
        target = (int(group), concentration)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    @staticmethod
    def targetTransform(target_index=0):
        def transform(target):
            target = target[target_index]
            target = torch.tensor(int(target))
            return target
        return transform
    
    @property
    def dataset(self):
        if self.dataset_file is None:
            self.dataset_file = h5py.File(self.data_path, 'r')
        if self.train:
            return self.dataset_file["BBBC013/train"]
        return self.dataset_file["BBBC013/test"]
    
    @property
    def images(self):
        if self.binary:
            return self.dataset['images'].value[self.dataset["binary_index_table"].value]
        return self.dataset['images'].value
    
    @property
    def groups(self):
        if self.binary:
            return self.dataset['groups'].value[self.dataset["binary_index_table"].value]
        return self.dataset['groups'].value
    
    @property
    def concentrations(self):
        if self.binary:
            return self.dataset['concentrations'].value[self.dataset["binary_index_table"].value]
        return self.dataset['concentrations'].value
    
    @property
    def targets(self):
        return self.groups, self.concentrations

    def download(self, data_path):
        """
        Download the BBBC013 dataset in data_path if it doesn't exist already.
        Images are augmented using aug_transform by a factor of n_augmentations and
        are cropped down to img_size.
        
        Args:
            data_path (string): Path to the dataset hdf5 file.
        """
        if os.path.exists(data_path):
            return
        
        print("[BBBC013] Loading images and metadata ...")
        metadata_DF = pd.read_csv(csv_file)
        metadata_DF = metadata_DF[metadata_DF.column != 2]
        data = np.load(img_file)
        print(data.shape)#.transpose(0, 3, 1, 2)
        
        print("[BBBC013] Processing training metadata ...")
        metadata_train_DF = metadata_DF[[r in ['A', 'B', 'C', 'E', 'F', 'G'] for r in metadata_DF.row]]
        concentration = metadata_train_DF.concentration.fillna(0).values.reshape(-1)
        group = metadata_train_DF.group.apply(lambda s: self.group_names.index(s)).values.reshape(-1)
        targets = np.array(list(zip(group, concentration)))

        selected_indexes = [i in metadata_train_DF.index for i in range(len(dataset))]
        selected_images = dataset[selected_indexes]

        print("[BBBC013] Augmenting training images ...")
        images = torch.empty(self.n_augmentations*len(selected_images), 2, self.img_size, self.img_size)
        labels = torch.empty(self.n_augmentations*len(selected_images), 2)
        for i in range(self.n_augmentations):
            images[i*len(selected_images):(i+1)*len(selected_images)] = torch.cat([self.aug_transform(img).view(1, 2, self.img_size, self.img_size) for img in selected_images])
            labels[i*len(selected_images):(i+1)*len(selected_images)] = torch.from_numpy(targets)
        
        print("[BBBC013] Saving training data at '{}' ...".format(data_path))
        with h5py.File(data_path, "w") as f:
            f.create_dataset("BBBC013/train/images", data=images.numpy())
            f.create_dataset("BBBC013/train/concentrations", data=labels.numpy()[:,1])
            f.create_dataset("BBBC013/train/groups", data=labels.numpy()[:,0].astype(np.uint8))
            f.create_dataset("BBBC013/train/binary_index_table", data=np.where([label[0] in [0, 1] for label in labels.numpy()])[0])
 
        print("[BBBC013] Processing test metadata ...")
        metadata_test_DF = metadata_DF[[r not in ['A', 'B', 'C', 'E', 'F', 'G'] for r in metadata_DF.row]]
        concentration = metadata_test_DF.concentration.fillna(0).values.reshape(-1)
        group = metadata_test_DF.group.apply(lambda s: groups.index(s)).values.reshape(-1)
        targets = np.array(list(zip(group, concentration)))

        selected_indexes = [i in metadata_test_DF.index for i in range(len(dataset))]
        selected_images = dataset[selected_indexes]
        
        print("[BBBC013] Augmenting test images ...")
        images = torch.empty(self.n_augmentations*len(selected_images), 2, self.img_size, self.img_size)
        labels = torch.empty(self.n_augmentations*len(selected_images), 2)
        for i in range(self.n_augmentations):
            images[i*len(selected_images):(i+1)*len(selected_images)] = torch.cat([self.aug_transform(img).view(1, 2, self.img_size, self.img_size) for img in selected_images])
            labels[i*len(selected_images):(i+1)*len(selected_images)] = torch.from_numpy(targets)
    
        print("[BBBC013] Saving test data at '{}' ...".format(data_path))
        with h5py.File(data_path, "w") as f:
            f.create_dataset("BBBC013/test/images", data=images.numpy())
            f.create_dataset("BBBC013/test/concentrations", data=labels.numpy()[:,1])
            f.create_dataset("BBBC013/test/groups", data=labels.numpy()[:,0].astype(np.uint8))
            f.create_dataset("BBBC013/test/binary_index_table", data=np.where([label[0] in [0, 1] for label in labels.numpy()])[0])
    
        print("[BBBC013] Done.")