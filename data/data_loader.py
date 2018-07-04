import torch
import transforms as trans

from torch.utils import data
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def load_data(data_root_path, batch_size, data_cfg, data_loader):
    data_transfer = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.RandomSampleCrop(data_cfg['resize_range'][0], data_cfg['resize_range'][1]),
        trans.Resize(data_cfg['crop_size']),
        trans.RandomHSV(),
        trans.ToTensor(),
        trans.Normalize(data_cfg['normal_mean'], data_cfg['normal_van'])
    ])

    dsets = data_loader(data_root_path, transforms=data_transfer)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=batch_size, shuffle=True, num_workers=8)
    dset_sizes = dsets.__len__()
    print "Total data num: %d" % dset_sizes
    return dset_loaders


def test_transfer(data_cfg):
    inference_transforms = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(data_cfg['normal_mean'],
                                                                    data_cfg['normal_van'])])
    return inference_transforms


class DetDataLoader(data.Dataset):

    def __init__(self):
        self.imgs = None
        self.scale = None
        self.is_gray = None
        self.data_path = None
        self.transforms = None
        self.num_classes = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item_idx):
        pass

    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
