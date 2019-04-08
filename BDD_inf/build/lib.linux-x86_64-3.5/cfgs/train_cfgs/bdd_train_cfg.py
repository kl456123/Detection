TrainCFG = {
    'lr': 0.0015,
    'lr_sch': [30000, 35000],
    'momentum': 0.9,
    'batch_size': 4,
    'test_fqc': 100,
    'eval_fqc': 1,
    'check_fqc': 1,
    'lr_multi': [1., 1.],
    'train_epochs': 15,
    'weight_decay': 0.0001,
    'devices': [0, 1, 2, 3],
    'criterion': 'os_loss',

    'is_fix_bn': False,
    'is_sync_bn': False,
    'is_parallel': False,
    'is_finetune': True,
    'is_print_loss': False,
    'use_adam_pretrain': False,

    'pretrained_weight': './weights/obj_epoch18.pkl',
    'is_local': False,
    'is_strict': True,

    'data_root': '/home/mark/Dataset/bdd100k',
    'train_path': 'images/100k/val',
    'train_anno': 'labels/bdd100k_val.json',
    'val_path': 'images/100k/val',
    'val_anno': 'labels/bdd100k_val.json',
    'save_path': './weights',
    'eval_out': './eval',
    'test_fig': './toy.jpg'
}

