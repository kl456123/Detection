TrainCFG = {
    'lr': 0.001,
    'lr_sch': [60000, 80000],
    'momentum': 0.9,
    'batch_size': 4,
    'test_fqc': 10,
    'eval_fqc': 1,
    'check_fqc': 1,
    'lr_multi': [1., 1.],
    'train_epochs': 12,
    'weight_decay': 0.0001,
    'devices': [0, 1, 2, 3],
    'criterion': 'os_loss',

    'is_fix_bn': False,
    'is_sync_bn': False,
    'is_parallel': False,
    'is_finetune': True,
    'is_print_loss': True,
    'use_adam_pretrain': False,

    'pretrained_weight': './weights/PRNet50_epoch49.pkl',
    'is_local': False,
    'is_strict': False,

    'data_root': '/home/mark/Dataset/KITTI/data_object_image_2/training',
    'train_path': 'image_2',
    'train_anno': 'kitti_train.json',
    'val_path': 'image_2',
    'val_anno': 'kitti_val.json',
    'save_path': './weights_kitti',
    'eval_out': './eval',
    'test_fig': './toy.png'
}

