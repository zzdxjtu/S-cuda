from data.eyes_dataset import eyessourceDataSet, eyestargetDataSet
from data.eyes_dataset_label import eyesDataSetLabel
import numpy as np
from torch.utils import data

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
image_sizes = {'train': (1024, 1024), 'val': (1024, 1024)}

def CreateSrcDataLoader(args): 
    source_dataset = eyessourceDataSet(args.data_dir, args.data_list,  args.load_selected_samples, crop_size=image_sizes['train'], mean=IMG_MEAN) 
    source_dataloader = data.DataLoader(source_dataset, batch_size=1, shuffle=False, pin_memory=True)
    return source_dataloader

def CreateTrgDataLoader(args):
    if args.data_label_folder_target is not None:
        target_dataset = eyesDataSetLabel(args.data_dir_target, args.data_list_target,
                                           max_iters=args.num_steps * args.batch_size,
                                           crop_size=image_sizes['val'], mean=IMG_MEAN,
                                           set=args.set, label_folder=args.data_label_folder_target) 
    else:
        if args.set == 'train':
            target_dataset = eyestargetDataSet(args.data_dir_target, args.data_list_target,
                                               max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['val'], mean=IMG_MEAN, set=args.set) 
        else:
            target_dataset = eyestargetDataSet(args.data_dir_target, args.data_list_target,
                                                crop_size=image_sizes['val'], mean=IMG_MEAN, set=args.set)             

    if args.set == 'train':
        target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    return target_dataloader

def CreateTrgDataSSLLoader(args):
    target_dataset = eyestargetDataSet(args.data_dir_target, args.data_list_target,
                                           crop_size=image_sizes['val'], mean=IMG_MEAN, set=args.set)
    target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)  
    return target_dataloader
