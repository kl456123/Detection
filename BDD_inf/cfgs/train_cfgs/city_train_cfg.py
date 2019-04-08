TrainCFG = {
    'lr': 0.0001,
    'lr_sch': [60000, 80000],
    'momentum': 0.9,
    'batch_size': 2,
    'test_fqc': 1000,
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

    'pretrained_weight': './weights/model_epoch18.pkl',
    'is_local': False,
    'is_strict': True,

    'data_root': '/home/mark/Dataset/Cityscapes',
    'train_path': 'leftImg8bit/train',
    'train_anno': 'det_anno/train.json',
    'val_path': 'images/100k/val',
    'val_anno': 'labels/bdd100k_val.json',
    'save_path': './weights',
    'eval_out': './eval',
    'test_fig': './toy.jpg'
}