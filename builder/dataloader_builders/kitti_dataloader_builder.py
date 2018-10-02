# -*- coding: utf-8 -*-

from data.datasets.kitti import KittiDataset

from builder.dataloader_builder import DataLoaderBuilder
import data.transforms.kitti_transform as trans


class KittiDataLoaderBuilder(DataLoaderBuilder):
    def build_dataset(self):
        """
        dataset_config, tranform_config and transform can be used
        """
        self.dataset = KittiDataset(self.dataset_config, self.transform,
                                    self.training)
        return self.dataset

    def build_transform(self):
        """
        tranform_config can be used
        """
        trans_cfg = self.tranform_config
        if self.training:
            self.transform = trans.Compose([
                trans.RandomHorizontalFlip(),
                trans.RandomSampleCrop(trans_cfg['resize_range'][0],
                                       trans_cfg['resize_range'][1]),
                trans.Resize(trans_cfg['crop_size']), trans.RandomHSV(),
                trans.ToTensor(), trans.Normalize(trans_cfg['normal_mean'],
                                                  trans_cfg['normal_van'])
            ])
        else:
            # Too ugly
            if trans_cfg.get('crop_size'):
                res = [trans.Resize(trans_cfg['crop_size'])]
            else:
                res = []
            res.append(trans.ToTensor())
            res.append(
                trans.Normalize(trans_cfg['normal_mean'],
                                trans_cfg['normal_van']))
            self.transform = trans.Compose(res)
        return self.transform
