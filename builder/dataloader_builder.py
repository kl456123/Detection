from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod


class DataLoaderBuilder(object, metaclass=ABCMeta):
    def __init__(self, data_config, training=True):
        self.tranform_config = data_config['transform_config']
        self.dataset_config = data_config['dataset_config']
        self.dataloader_config = data_config['dataloader_config']
        # phase
        self.training = training

        self.transform = None
        self.dataset = None
        self.dataloader = None

    @abstractmethod
    def build_transform(self):
        pass

    @abstractmethod
    def build_dataset(self):
        pass

    def build_dataloader(self):
        config = self.dataloader_config
        batch_size = config['batch_size']
        shuffle = config['shuffle']
        num_workers = config['num_workers']
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
        return self.dataloader

    def build(self):
        self.build_transform()
        self.build_dataset()
        self.build_dataloader()
        return self.dataloader
