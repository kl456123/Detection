TrainCFG = {
    'lr': 0.001,
    'lr_sch': [60000, 80000],
    'momentum': 0.9,
    'batch_size': 8,
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
    'is_finetune': False,
    'is_print_loss': False,
    'use_adam_pretrain': False,

    'pretrained_weight': './weights/PRNet50_epoch49.pkl',
    'is_local': False,
    'is_strict': True,

    'data_root': '/home/mark/Dataset/COCO',
    'train_path': 'val2017',
    'train_anno': 'annotations/instances_val2017.json',
    'val_path': 'val2017',
    'val_anno': 'annotations/instances_val2017.json',
    'save_path': './weights',
    'eval_out': './eval',
    'test_fig': './toy.jpg'
}

