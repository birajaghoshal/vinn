import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from scipy.misc import imread

# Creating a sub class of torch.utils.data.dataset.Dataset
class notMNIST(Dataset):

    # The init method is called when this class will be instantiated.
    def __init__(self, root):
        Images, Y = [], []
        folders = os.listdir(root)

        for folder in folders:
            folder_path = os.path.join(root, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    Images.append(np.array(imread(img_path)))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    # Some images in the dataset are damaged
                    print("File {}/{} is broken".format(folder, ims))
        data = [(x, y) for x, y in zip(Images, Y)]
        self.data = data
        self.labels = np.array(Y)
        self.images = np.array(Images)

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    # The Dataloader is a generator that repeatedly calls the getitem method.
    # getitem is supposed to return (X, Y) for the specified index.
    def __getitem__(self, index):
        img = self.data[index][0]

        # 8 bit images. Scale between [0,1]. This helps speed up our training
        img = img.reshape(28, 28) / 255.0

        # Input for Conv2D should be Channels x Height x Width
        img_tensor = torch.Tensor(img).view(1, 28, 28).float()
        label = self.data[index][1]
        return (img_tensor, label)