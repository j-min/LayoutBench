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

# from datasets.coco import CocoDetection, make_coco_transforms, convert_coco_poly_to_mask
from datasets.coco import CocoDetection, convert_coco_poly_to_mask

from copy import deepcopy
import random
import os

class CLEVRDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, args=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.return_masks = return_masks
        self.args = args

        self.prepare = ConvertCocoPolysToMask(return_masks, args=args)

        self.catid2name = {} 
        for catid in self.coco.cats:
            self.catid2name[catid] = self.coco.cats[catid]['name']

        self.img_folder = img_folder
        self.ann_file = ann_file

        print('========================')
        print('# of images: ', len(self.ids))
        print('========================')

        if args.gtshuffle:
            print('========================')
            print('GT Shuffle mode is enabled')
            print('========================')

    def _load_image(self, id: int) -> Image.Image:
        if self.args.gtshuffle:
            # Load a random image id
            id = random.choice(self.ids)
            path = self.coco.loadImgs(id)[0]["file_name"]
            return Image.open(os.path.join(self.root, path)).convert("RGB")
        else:
            # Original
            path = self.coco.loadImgs(id)[0]["file_name"]
            return Image.open(os.path.join(self.root, path)).convert("RGB")

        

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)

        orig_img = img

        # if self.args.dataset_file in ['clevr320', 'layoutbench_v1']:
        
        # Resize to 320x320
        # because their annotations are designed for 320x320
        img = img.resize((320, 320), Image.BICUBIC)

        # if self.args.layoutbench_resized_image_eval:
        #     assert img.size == (512, 512), f"Loaded image size should be 512x512, but got {img.size}"
        #     # loaded image is already resized to 512x512, while target is designed for 320x320
        #     img = img.resize((320, 320), Image.BICUBIC)
        # if self.args.clevr_resized_image_eval:
            

        #     # Original: 480x320 (WxH)
        #     # -> resized to 768x512
        #     # -> center cropped to 512x512
        #     # Let's do reverse

        #     # Loaded size: 512x512
        #     assert img.size == (512, 512), f"Loaded image size should be 512x512, but got {img.size}"
        #     # # pad left and right to 768x512
        #     # def add_margin(pil_img, top, right, bottom, left, color):
        #     #     width, height = pil_img.size
        #     #     new_width = width + right + left
        #     #     new_height = height + top + bottom
        #     #     result = Image.new(pil_img.mode, (new_width, new_height), color)
        #     #     result.paste(pil_img, (left, top))
        #     #     return result
        #     # img = add_margin(img, 0, 128, 0, 128, (0, 0, 0))

        #     # # resize to 480x320 (WxH)
        #     # img = img.resize((480, 320), Image.BICUBIC)

        #     # bounding box target should not include padding
        #     # original bbox target is designed for 480x320
        #     # so we need to 1) resize annotation to 768x512, 2) center crop to 512x512

        #     img = img.resize((320, 320), Image.BICUBIC)

        #     # # 1) resize annotation from 480x320 to 768x512
        #     # target = deepcopy(target)

        #     # print('bbox before:', target[0]['bbox'])

        #     # for obj in target:
        #     #     # bbox: [x, y, w, h]
        #     #     bbox = obj['bbox']
        #     #     bbox[0] *= 1.6
        #     #     bbox[1] *= 1.6
        #     #     bbox[2] *= 1.6
        #     #     bbox[3] *= 1.6

        #     # print('bbox after resize (x1.6):', target[0]['bbox'])

        #     # # 2) center crop annotation to 512x512
        #     # for obj in target:
        #     #     # bbox: [x, y, w, h]
        #     #     bbox = obj['bbox']

        #     #     x0 = bbox[0]
        #     #     w = bbox[2]
        #     #     x1 = x0 + w
                
        #     #     # center crop (only y axis)
        #     #     x0 -= 128
        #     #     x1 -= 128

        #     #     x0 = max(x0, 0)
        #     #     x1 = min(x1, 512)
        #     #     w = x1 - x0

        #     #     bbox[0] = x0
        #     #     bbox[2] = w

        #     # print('bbox after center crop:', target[0]['bbox'])

        transformed_img = img

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        target['orig_img'] = orig_img
        target['transformed_img'] = transformed_img

        filename = self.coco.loadImgs(image_id)[0]["file_name"]
        img_path = Path(self.root) / filename
        target['img_path'] = img_path
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

        return image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),

        # box_xyxy_to_cxcywh
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            # box_xyxy_to_cxcywh
            normalize,
        ])

    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),

            # box_xyxy_to_cxcywh
            normalize,
        ])

    # elif image_set == 'centercrop_test':
    #     print("Using center crop for CLEVR test set for DETR-based evaluation")
    #     return T.Compose([
    #         T.RandomResize([512]),
    #         T.CenterCrop((512, 512)),
    #         # box_xyxy_to_cxcywh
    #         normalize,
    #     ])

    raise ValueError(f'unknown {image_set}')


def convert_tensor_to_pil(tensor):
    """Reverse transform """

    assert tensor.dim() == 3, "Input tensor should be (C, H, W)"

    tensor = tensor.clone().detach().cpu()  # avoid modifying the input tensor

    # reverse normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean

    # reverse ToTensor
    from torchvision.transforms import ToPILImage
    img = ToPILImage()(tensor)

    return img

def build_clevr_dataset(image_set,  args):

    root = Path(args.clevr_dir)
    assert root.exists(), f'provided path {root} does not exist'

    if args.dataset_file == 'clevr':
        img_folder = root / "CLEVR_v1.0/images" / image_set

        if image_set == 'train':
            ann_file = root / 'coco_format' / f"train.json"
        elif image_set == 'val':
            if args.clevr_resized_image_eval:
                ann_file = root / 'coco_format' / f"val_320x320.json"
            else:
                ann_file = root / 'coco_format' / f"val.json"

    elif args.dataset_file == 'clevr320':

        # scene_path = clevr_dir / split / 'scenes.json'
        # with open(scene_path) as f:
        #     scenes = json.load(f)
        # print('Loaded ', scene_path, ' | shape: ', len(scenes['scenes']))

        # image_dir = clevr_dir / split / 'images'

        
        img_folder = root / image_set / 'images'
        ann_file = root / image_set / f"scenes_coco.json"

    # if args.clevr_resized_image_eval:
    #     transform = make_coco_transforms('centercrop_test')
    # else:
    transform = make_coco_transforms(image_set)

    if args.eval_image_dir is not None:
        img_folder = Path(args.eval_image_dir)
        print(f'using eval image dir: {img_folder}')

    dataset = CLEVRDataset(img_folder, ann_file,
                                transforms=transform, return_masks=args.masks,
                                args=args)

    return dataset

def build_layoutbench_dataset(image_set, args):

    root = Path(args.layoutbench_dir)
    assert root.exists(), f'provided path {root} does not exist'


    if args.dataset_file == 'layoutbench_v1':
        skill, subsplit = args.skill_split.split('_', 1)

        if image_set == 'train':
            skill_dir = Path(args.layoutbench_dir) / args.train_set / image_set / skill
        elif image_set == 'val':
            skill_dir = Path(args.layoutbench_dir) / args.val_set / image_set / skill

        ann_file = skill_dir / 'coco' / f'scenes_{args.skill_split}_coco.json'
        img_folder = skill_dir / "images"

    elif args.dataset_file == 'layoutbench_v0':
        assert args.skill_split in [
            'number_few', 'number_many',
            'position_boundary', 'position_center',
            'size_tiny', 'size_verylarge',
            'shape_horizontal', 'shape_vertical',
            'allskills_allsplits'
        ], f'invalid skill split {args.skill_split}'

        skill, split = args.skill_split.split('_')

        if image_set == 'train':
            skill_split_dir = Path(args.layoutbench_dir) / args.train_set / skill / split 
        elif image_set == 'val':
            skill_split_dir = Path(args.layoutbench_dir) / args.val_set / skill / split

        ann_file = skill_split_dir / f'scenes_coco.json'

        img_folder = skill_split_dir / "images"
    
    if args.eval_image_dir is not None:
        img_folder = Path(args.eval_image_dir)
        print('==============================')
        print(f'using eval image dir: {img_folder}')
        print('==============================')

    # if args.layoutbench_resized_image_eval and image_set == 'val':
    #     transform = make_coco_transforms('centercrop_test')
    # else:
    transform = make_coco_transforms(image_set)

    dataset = CLEVRDataset(img_folder, ann_file,
                                transforms=transform, return_masks=args.masks,
                                args=args)

    return dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--clevr_dir', type=str, default='/datadrive_a/jaemin/datasets/clevr_data/')
    args = parser.parse_args()

    dataset = build_clevr_dataset('train', args)
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


