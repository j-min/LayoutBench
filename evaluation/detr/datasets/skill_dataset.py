from pathlib import Path

import torch
import torch.utils.data
import torchvision
# from pycocotools import mask as coco_mask

import datasets.transforms as T
import json
from PIL import Image

from string import digits
remove_digits = str.maketrans('', '', digits)

from datasets.coco import CocoDetection, make_coco_transforms, convert_coco_poly_to_mask

class PaintSkillDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, metadata_file, transforms, return_masks, args=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.return_masks = return_masks
        self.args = args

        self.metadata = json.load(open(metadata_file))

        # self.class_name_to_id = {name: id for id, name in enumerate(self.metadata['Shape'])}
        # self.color_name_to_id = {name: id for id, name in enumerate(self.metadata['Color'])}

        # print('color name2id')
        # print(self.color_name_to_id)

        self.prepare = ConvertCocoPolysToMask(return_masks, args=args)
        # self.prepare.color_name_to_id = self.color_name_to_id
        # self.prepare.class_name_to_id = self.class_name_to_id

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, args=None):
        self.return_masks = return_masks
        self.args = args

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # ##### Added #####
        # # if self.args is not None and self.args.num_colors > 0:
        # if self.args is not None and self.args.num_colors > 0 and self.args.skill_name in ['color', 'overall']:
        #     if 'color' in anno[0] and anno[0]['color'] not in [None, "plain"]:
        #         colors = []
        #         for obj in anno:
        #             color_name = obj["color"]
        #             color_id = self.color_name_to_id[color_name]
        #             colors.append(color_id)
        #         colors = torch.tensor(colors, dtype=torch.int64)
        #         colors = colors[keep]
        #     else:
        #         colors = [-100] * len(boxes)
        #         colors = torch.tensor(colors, dtype=torch.int64)
        #     target['colors'] = colors
        # else:
        #     colors = [-100] * len(boxes)
        #     colors = torch.tensor(colors, dtype=torch.int64)
        #     target['colors'] = colors
        # #################

        return image, target

def build_dataset(image_set,  args):

    root = Path(args.paintskills_dir)

    assert root.exists(), f'provided path {root} does not exist'

    # if args.skill_name == 'allskills':
    #     skill_names = ['object', 'count', 'spatial']
    #     print('skill_names', skill_names)
    #     from copy import deepcopy
    #     from torch.utils.data import ConcatDataset
    #     datasets = []
    #     for skill in skill_names:
    #         _args = deepcopy(args)
    #         _args.skill_name = skill
    #         dataset = build_dataset(image_set, _args)
    #         datasets.append(dataset)
    #     dataset = ConcatDataset(datasets)

    # else:

    skill_dir = root / args.skill_name
    print('Loading dataset from {}'.format(skill_dir))

    img_folder = skill_dir / "images"
    ann_file = skill_dir / f"{args.skill_name}_{image_set}_bounding_boxes.json"
    metadata_file = root / "metadata.json"

    dataset = PaintSkillDataset(img_folder, ann_file, metadata_file=metadata_file,
                                transforms=make_coco_transforms(image_set), return_masks=args.masks,
                                args=args)

    return dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--skill_name', type=str, default='object')
    args = parser.parse_args()


    dataset = build_dataset('train', args)
    print(dataset)

    from tqdm import tqdm

    import util.misc as utils

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
        )

    for batch in tqdm(data_loader):
        print(batch)


