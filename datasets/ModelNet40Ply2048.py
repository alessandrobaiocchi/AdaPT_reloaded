import glob
import os
from typing import List, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40Ply2048(Dataset):
    def __init__(
        self,
        root,
        split="train",
    ):
        assert split == "train" or split == "test"

        self.split = split
        self.data_list = []
        self.labels_list = []
        for h5_name in glob.glob(os.path.join(root, "ply_data_%s*.h5" % split)):
            with h5py.File(h5_name, "r") as f:
                self.data_list.append(f["data"][:].astype(np.float32))  # type: ignore
                self.labels_list.append(f["label"][:].astype(np.int64))  # type: ignore
        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0).squeeze(-1)

    def __getitem__(self, item):
        points = self.data[item]
        label = self.labels[item]
        if self.split == 'train':
            points = random_point_dropout(points) # open for dgcnn not for our idea  for all
            points = translate_pointcloud(points)
            np.random.shuffle(points)
        return points, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Ply2048DataModule(pl.LightningDataModule):
    """
    size: 12308
    train: 9840
    test: 2468
    """

    def __init__(
        self,
        data_dir: str = "/home/alebai/projects/Datasets/PointClouds/modelnet40_ply_hdf5_2048",
        batch_size: int = 32,
        drop_last: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        with open(os.path.join(data_dir, "shape_names.txt"), "r") as f:
            self._label_names = [line.rstrip() for line in f]

    def setup(self, stage: Optional[str] = None):
        self.modelnet_train = ModelNet40Ply2048(self.hparams.data_dir, split="train")  # type: ignore
        self.modelnet_test = ModelNet40Ply2048(self.hparams.data_dir, split="test")  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.modelnet_train,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=self.hparams.drop_last,  # type: ignore
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.modelnet_test,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=4,
        )

    @property
    def num_classes(self):
        return 40

    @property
    def label_names(self) -> List[str]:
        return self._label_names
    
    @property
    def num_points(self):
        return 2048
