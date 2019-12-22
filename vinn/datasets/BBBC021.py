import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import h5py
        
class BBBC021(Dataset):
    """BBBC021 dataset <https://data.broadinstitute.org/bbbc/BBBC021/>"""

    csv_file = "/da/isld/share/deeplearning/datasets/public/BBBC021/metadata.csv"
    img_file = "/da/isld/share/deeplearning/datasets/public/BBBC021/images.npy"
    
    MOA_names = ['Actin disruptors', 'Aurora kinase inhibitors', 'Cholesterol-lowering',
                 'DNA damage', 'DNA replication', 'Eg5 inhibitors', 'Epithelial', 'DMSO',
                 'Kinase inhibitors', 'Microtubule destabilizers', 'Microtubule stabilizers',
                 'Protein degradation', 'Protein synthesis']
    
    compound_names = ['ALLN', 'AZ-A', 'AZ-C', 'AZ-J', 'AZ-U', 'AZ138', 'AZ258', 'AZ841', 'DMSO',
                         'MG-132', 'PD-169316', 'PP-2', 'alsterpaullone', 'anisomycin', 'bryostatin',
                         'camptothecin', 'chlorambucil', 'cisplatin', 'colchicine', 'cyclohexamide',
                         'cytochalasin B', 'cytochalasin D', 'demecolcine', 'docetaxel', 'emetine',
                         'epothilone B', 'etoposide', 'floxuridine', 'lactacystin', 'latrunculin B',
                         'methotrexate', 'mevinolin/lovastatin', 'mitomycin C', 'mitoxantrone',
                         'nocodazole', 'proteasome inhibitor I', 'simvastatin', 'taxol', 'vincristine']

    mean = [0.30828, 0.179584, 0.420805]
    std = [1.57484, 1.49378, 1.95447]
    perc07 = [-0.86572266, -1.08496094, -0.1776123]
    perc93 = [2.60351562, 2.375, 2.5625]
    
    def __init__(self, LOCO=None, LOMOAO=None, train=True, download=False, transform=None, target_transform=None, data_path="data/BBBC021/BBBC021.hdf5"):
        """
        Args:
            LOCO (string, optional): If None, the entire dataset is used (check LOMOAO),
                otherwise the Leave-One-Compound-Out approach is adopted to select training
                and validation sets.
            LOMOAO (string, optional): Same as LOCO but excludes an MOA instead of a compound.
            train (bool, optional): If LOCO is None it has no effect, otherwise, if 
                True, creates the dataset excluding the compound specified in LOCO. Else
                it uses only data corresponding to the compound specified in LOCO.
            download (bool, optional): If true, downloads the dataset from the Isilon to
                data_path using csv_file and img_file for the metadata and the images 
                respectively.
                If dataset is already downloaded, it is not downloaded again.
            transform (callable, optional): A function/transform to be applied on a sample
                numpy image with format [H, W, C].
            target_transform (callable, optional): A function/transform to be applied on a
                sample target with the format (MOA, compound, concentration).
            data_path (string, optional): Path to the dataset hdf5 file.
        """ 
        
        if download:
            self.download(data_path)

        if not os.path.exists(data_path):
            raise RuntimeError("Dataset not found at '{}'. Use download=True to download it.".format(data_path))
            
        self.LOCO = LOCO
        self.LOMOAO = LOMOAO
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_path = data_path
        self.dataset_file = None
        self.n_items = None
        
        with h5py.File(self.data_path, 'r') as dataset_file:
            dataset = dataset_file['BBBC021']
            if self.LOCO is not None:
                if self.train:
                    self.index_table = np.where([self.compound_names[compound] != self.LOCO for compound in dataset['compounds']])[0]
                else:
                    self.index_table = np.where([self.compound_names[compound] == self.LOCO for compound in dataset['compounds']])[0]
                self.n_items = self.index_table.shape[0]
            elif self.LOMOAO is not None:
                if self.train:
                    self.index_table = np.where([self.MOA_names[moa] != self.LOMOAO for moa in dataset['MOAs']])[0]
                else:
                    self.index_table = np.where([self.MOA_names[moa] == self.LOMOAO for moa in dataset['MOAs']])[0]
                self.n_items = self.index_table.shape[0]
            else:
                self.n_items = dataset["images"].shape[0]
        
    def __len__(self):           
        return self.n_items

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, (MOA, compound, concentration)) where MOA and compound are the indexes in
                        MOA_names and compound_names respectively.
        """        
        if self.LOCO is not None or self.LOMOAO is not None:
            index = self.index_table[index]
        
        img = self.dataset['images'][index].astype(np.float32).transpose(1, 2, 0)
        compound = self.dataset['compounds'][index]
        concentration = self.dataset['concentrations'][index]
        moa = self.dataset['MOAs'][index]
        target = (int(moa), int(compound), concentration)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    @property
    def dataset(self):
        if self.dataset_file is None:
            self.dataset_file = h5py.File(self.data_path, 'r')
        return self.dataset_file["BBBC021"]        
    
    @property
    def images(self):
        if self.LOCO is not None or self.LOMOAO is not None:
            return self.dataset['images'].value[self.index_table].astype(np.float32).transpose(0, 2, 3, 1)
        return self.dataset['images'].value.astype(np.float32).transpose(0, 2, 3, 1)
    
    @property
    def compounds(self):
        if self.LOCO is not None or self.LOMOAO is not None:
            return self.dataset['compounds'].value[self.index_table]
        return self.dataset['compounds'].value
    
    @property
    def concentrations(self):
        if self.LOCO is not None or self.LOMOAO is not None:
            return self.dataset['concentrations'].value[self.index_table]
        return self.dataset['concentrations'].value
    
    @property
    def MOAs(self):
        if self.LOCO is not None or self.LOMOAO is not None:
            return self.dataset['MOAs'].value[self.index_table]
        return self.dataset['MOAs'].value
    
    @property
    def targets(self):
        if self.target_transform is None:
            return self.MOAs, self.compounds, self.concentrations
        targets = [self.target_transform([t1, t2, t3]) for t1, t2, t3 in zip(self.MOAs, self.compounds, self.concentrations)]
        return targets
    
    @staticmethod
    def transform(image):
        print('UPDATE!')
    
    @staticmethod
    def targetTransform(moa=None):
        if moa is None:
            return lambda x: torch.tensor(int(x[0]))
        
        new_target = BBBC021.MOA_names.index(moa)

        def transform(target):
            target = target[0]
            if target > 11:
                target = new_target
            target = torch.tensor(int(target))
            return target
        return transform
    
    @staticmethod
    def show(image, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        
        ax.imshow(self.transform(image))
        ax.axis('off')
    
    def download(self, data_path):
        """Download the BBBC021 dataset in data_path if it doesn't exist already.
        
        Args:
            data_path (string): Path to the dataset hdf5 file.
        """
        if os.path.exists(data_path):
            return
        
        print("[BBBC021] Loading images and metadata ...")
        metadata_DF = pd.read_csv(self.csv_file).drop(["plate", "row", "column", "table", "number", "replicate"], axis=1)
        images = np.load(self.img_file).transpose(0, 3, 1, 2)
        
        print("[BBBC021] Processing metadata ...")
        #toDrop = [0, 2, 6, 8, 11]
        #for moa_index in toDrop:
        #    metadata_DF = metadata_DF.drop(labels=metadata_DF[(metadata_DF.moa == self.MOA_names[moa_index])].index, axis=0)
        median_len = int(np.median(np.unique(metadata_DF.moa, return_counts=True)[1]))
        metadata_DF = metadata_DF.drop(labels=metadata_DF[(metadata_DF.moa == 'DMSO')].index[median_len:], axis=0)
        metadata_DF = metadata_DF.drop(labels=metadata_DF[(metadata_DF.moa == 'Microtubule stabilizers')].index[median_len:], axis=0)
        images = images[[i in metadata_DF.index.tolist() for i in range(len(images))]]
        
        MOAs = metadata_DF['moa'].apply(lambda s: self.MOA_names.index(s)).values.reshape(-1).astype(np.uint8)
        compounds = metadata_DF['compound'].apply(lambda s: self.compound_names.index(s)).values.reshape(-1).astype(np.uint8)
        concentrations = metadata_DF['concentration'].values.reshape(-1)
        
        print("[BBBC021] Saving dataset at '{}' ...".format(data_path))
        with h5py.File(data_path, "w") as f:
            f.create_dataset("BBBC021/images", data=images)
            f.create_dataset("BBBC021/MOAs", data=MOAs)
            f.create_dataset("BBBC021/compounds", data=compounds)
            f.create_dataset("BBBC021/concentrations", data=concentrations)
        
        print("[BBBC021] Done.")