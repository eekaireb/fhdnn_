from argparse import ArgumentParser
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from SeqSampler import SeqSampler
import pytorch_lightning as pl

class MnistData(pl.LightningDataModule):
    def __init__(self, root = './codeden/data/mnist', batch_size = 512, workers = 8, debug = False, expt = 'cnn', **kw):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.debug = debug
        self.root = root

        self.train_samples_per_cls = 30000
        self.test_samples_per_cls = 500
        self.training_data_type = 'sequential'
        self.blend_ratio = 0
        self.n_concurrent_classes = 1
        self.train_sampler = None
        
        if expt == 'cnn':
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.train_transform = transforms.Compose([
                #
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                #
                transforms.ToTensor()
            ])

            self.val_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            self.flip_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            self.train_transform_runtime = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])


    def trainset(self):
        return MNIST(root=self.root, train=True, download = True, transform = self.train_transform)
    
    def valset(self):
        return MNIST(root=self.root, train=False, download = True, transform = self.val_transform)

    def flipset(self):
        return MNIST(root=self.root, train=False, download = True, transform = self.flip_transform)

    @staticmethod
    def add_loader_specific_args(parent_parser):
        parent_parser.add_argument('-b', '--batch_size', type = int, default = 512)
        parent_parser.add_argument('--workers', type = int, default = 8)
        parent_parser.add_argument('--debug', type = int, default = 0)
        parent_parser.add_argument('--mode', type = int, default = 10)

        return parent_parser
    
    def make_loader(self, split):
        if split == 'train':
            dataset = self.trainset()
            labels = dataset.targets.detach().cpu().numpy()
            num_labels = len(list(set(labels)))
            train_subset_len = num_labels * self.train_samples_per_cls

            train_subset, _ = torch.utils.data.random_split(dataset=dataset,
                                                        lengths=[train_subset_len,
                                                        len(dataset) - train_subset_len])
            print("train data")
            print(train_subset)
            print(len(train_subset))
            self.train_sampler = SeqSampler(train_subset, self.blend_ratio, self.n_concurrent_classes) \
                if self.training_data_type == 'sequential' else None
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=(self.train_sampler is None),
                            num_workers=self.workers, pin_memory=True, sampler=self.train_sampler)

            return train_loader
            #
        elif split == 'val':
            dataset = self.valset()
            val_subset_len = num_labels * self.test_samples_per_cls
            val_subset, _ = DataLoader.random_split(dataset=dataset,
                                                    lengths=[val_subset_len, 
                                                    len(dataset) - val_subset_len])
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
            return val_loader
        elif split == 'flip':
            dataset = self.flipset()
        else:
            dataset = self.valset()

        loader = DataLoader(dataset, batch_size = self.batch_size, num_workers = self.workers)
        
        return loader

    def client_loader(self, client_data):
        train_loader = DataLoader(client_data, batch_size=self.batch_size, shuffle=(self.train_sampler is None),
                            num_workers=self.workers, pin_memory=True, sampler=self.train_sampler)
        return train_loader
    
    def train_dataloader(self):
        return self.make_loader(split='train')
    
    def val_dataloader(self):
        return self.make_loader(split='val')
    
    def test_dataloader(self):
        return self.make_loader(split='val')

    def flip_dataloader(self):
        return self.make_loader(split='flip')
