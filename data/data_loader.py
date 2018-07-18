import torch
import data.bev_transform as trans

from torch.utils import data
from torchvision import transforms



def load_data(batch_size, data_cfg, encoder, data_loader, save_datafile=False):
    data_transfer = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
    ])

    dsets = data_loader(data_cfg, encoder, transforms=data_transfer)
    #ipdb.set_trace()
    #dsets[0]
    dset_loaders = torch.utils.data.DataLoader(
        dsets, batch_size=batch_size, shuffle=True, num_workers=8)
    dset_sizes = dsets.__len__()
    print(("Total data num: %d" % dset_sizes))
    return dset_loaders


def test_transfer(data_cfg):
    inference_transforms = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(data_cfg['normal_mean'],
                                                    data_cfg['normal_van'])
    ])
    return inference_transforms



