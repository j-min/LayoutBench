# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .coco import build_coco30k
from .coco import build_coco5k
# from .skill_dataset import build_dataset as build_skill_dataset

from .clevr import build_clevr_dataset, build_layoutbench_dataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

    # if isinstance(dataset, torch.utils.data.ConcatDataset):
    #     datasets = dataset.datasets
    #     for dataset in datasets:
    #         coco_api = get_coco_api_from_dataset(dataset)
    #         if coco_api is not None:
    #             return coco_api


def build_dataset(image_set, args):

    print('args.dataset_file', args.dataset_file)

    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    elif args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    
    elif args.dataset_file == 'coco30k':
        return build_coco30k(image_set, args)
    
    elif args.dataset_file == 'coco_val2017':
        return build_coco5k(image_set, args)
        
    elif 'clevr' in args.dataset_file:
        return build_clevr_dataset(image_set, args)
    elif 'layoutbench' in args.dataset_file:
        return build_layoutbench_dataset(image_set, args)
    
    raise ValueError(f'dataset {args.dataset_file} not supported')
