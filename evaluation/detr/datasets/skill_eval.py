# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Skill evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/skill_eval.py
The difference is that there is less copy-pasting from pyskilltools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

import json

from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from copy import deepcopy
from PIL import Image
from tqdm import tqdm

from torchvision import transforms as T

def paintskills_object_to_coco_names(obj):
    if 'human' in obj:
        return "person"
    elif "bike" == obj:
        return "bicycle"
    elif "fireHydrant" == obj:
        return "fire hydrant"
    elif "stopSign" == obj:
        return "stop sign"
    elif "trafficLight" == obj:
        return "traffic light"
    elif "pottedPlant" == obj:
        return "potted plant"
    return obj


# COCO classes
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class PaintSkillsDETREvaluationDataset(Dataset):
    def __init__(self, image_dir, ann_path, metadata_path, args=None):
        self.args = args

        self.image_dir = image_dir
        self.ann_data = json.load(open(ann_path))
        self.metadata = json.load(open(metadata_path))

        if args.human_eval_50:
            print('Skill: {}, len: {}'.format(args.skill_name, len(self.ann_data['data'])))
            if args.skill_name == 'object':
                amt_50_ids = ['object_val_01807', 'object_val_02245', 'object_val_02192', 'object_val_01760', 'object_val_00368', 'object_val_01190', 'object_val_01614', 'object_val_00566', 'object_val_01351', 'object_val_01419', 'object_val_01497', 'object_val_00659', 'object_val_00927', 'object_val_01602', 'object_val_00688', 'object_val_00697', 'object_val_00789', 'object_val_01466', 'object_val_01703', 'object_val_00242', 'object_val_02309', 'object_val_00929', 'object_val_00652', 'object_val_02039', 'object_val_00770',
                    'object_val_02075', 'object_val_01408', 'object_val_01877', 'object_val_02250', 'object_val_02056', 'object_val_01721', 'object_val_01567', 'object_val_01685', 'object_val_00275', 'object_val_01562', 'object_val_00260', 'object_val_01012', 'object_val_02273', 'object_val_00857', 'object_val_01143', 'object_val_01108', 'object_val_02078', 'object_val_00764', 'object_val_00849', 'object_val_00922', 'object_val_01280', 'object_val_00201', 'object_val_01276', 'object_val_00412', 'object_val_00915']
            elif args.skill_name == 'count':
                amt_50_ids = ['count_val_00657', 'count_val_00481', 'count_val_00309', 'count_val_01645', 'count_val_01618', 'count_val_00249', 'count_val_01878', 'count_val_01088', 'count_val_01328', 'count_val_01841', 'count_val_01123', 'count_val_01348', 'count_val_01416', 'count_val_01099', 'count_val_01477', 'count_val_01084', 'count_val_01486', 'count_val_00153', 'count_val_01924', 'count_val_00542', 'count_val_01884', 'count_val_00291', 'count_val_01273', 'count_val_01318', 'count_val_01041', 'count_val_00538', 'count_val_00929', 'count_val_01360', 'count_val_01752', 'count_val_00945', 'count_val_00034', 'count_val_00572', 'count_val_02131', 'count_val_01850', 'count_val_00495', 'count_val_00029', 'count_val_00253', 'count_val_02151', 'count_val_01152', 'count_val_00489', 'count_val_01208', 'count_val_01852', 'count_val_01529', 'count_val_00340', 'count_val_00472', 'count_val_00655', 'count_val_00135', 'count_val_00061', 'count_val_00177', 'count_val_00437']
            elif args.skill_name == 'spatial':
                amt_50_ids = ['spatial_val_01528', 'spatial_val_00926', 'spatial_val_01943', 'spatial_val_00959', 'spatial_val_00680', 'spatial_val_00903', 'spatial_val_00245', 'spatial_val_00576', 'spatial_val_00087', 'spatial_val_00293', 'spatial_val_02345', 'spatial_val_01887', 'spatial_val_02534', 'spatial_val_02300', 'spatial_val_00592', 'spatial_val_02558', 'spatial_val_01991', 'spatial_val_00601', 'spatial_val_00128', 'spatial_val_00148', 'spatial_val_02158', 'spatial_val_00983', 'spatial_val_02489', 'spatial_val_02217', 'spatial_val_02460', 'spatial_val_02058', 'spatial_val_00110', 'spatial_val_00322', 'spatial_val_00952', 'spatial_val_00275', 'spatial_val_01335', 'spatial_val_00173', 'spatial_val_02030', 'spatial_val_00960', 'spatial_val_00664', 'spatial_val_02569', 'spatial_val_00130', 'spatial_val_01205', 'spatial_val_01496', 'spatial_val_00020', 'spatial_val_01783', 'spatial_val_01985', 'spatial_val_01447', 'spatial_val_02440', 'spatial_val_02331', 'spatial_val_02101', 'spatial_val_02455', 'spatial_val_02441', 'spatial_val_01659', 'spatial_val_01055']
            data = []
            for datum in self.ann_data['data']:
                if datum['id'] in amt_50_ids:
                    data.append(datum)
            self.ann_data = {'data': data}
            print('Using AMT ids, len: {}'.format(len(self.ann_data['data'])))

        self.img_transform = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        self.shape_to_ix = {shape: i for i, shape in enumerate(COCO_CLASSES) if shape != 'N/A'}
        self.ix_to_shape = {i: shape for shape, i in self.shape_to_ix.items()}

        self.ix_paintskills = []
        for shape in self.metadata['Shape']:
            # print(shape, '->', paintskills_object_to_coco_names(shape))
            self.ix_paintskills.append(self.shape_to_ix[paintskills_object_to_coco_names(shape)])

    def __len__(self):
        return len(self.ann_data['data'])

class ObjectDataset(PaintSkillsDETREvaluationDataset):

    def __getitem__(self, ix):
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)

        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)

        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['target'] = []
        out['img_ids'] = []
        for datum in batch:
            shape = datum['objects'][0]['shape']
            if 'human' in shape:
                shape = 'human'
            # if not self.args.FT:
            shape = paintskills_object_to_coco_names(shape)
            shape_id = self.shape_to_ix[shape]
            out['target'].append(shape_id)
            out['img_ids'].append(datum['img_id'])

        out['target'] = torch.LongTensor(out['target'])

        return out


def eval_object(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []
    obj_specific_results = {}

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        pred_logits = outputs['pred_logits']

        if args.ignore_other_classes:
            for cls_id in range(pred_logits.size(-1)):
                if cls_id not in dataset.ix_paintskills:
                    pred_logits[:, :, cls_id] = -1e10

        # [B, n_queries, num_classes]
        all_probas = pred_logits.softmax(-1)[:, :, :-1]

        # [B, num_classes]
        probas_max_query = all_probas.max(1).values

        # pred_id = probas.max(1).values.max(-1).indices

        # [B]
        pred_prob = probas_max_query.max(-1).values
        pred_id = probas_max_query.max(-1).indices

        # keep only predictions with confidence
        # pred_id[pred_prob < args.p_threshold] = -1

        target = batch["target"].to(device)

        # print('probas:', probas)
        # print('pred_id:', pred_id)
        # print('target:', target)

        # break

        correct = pred_id == target
        # correct = correct.float()

        total += B
        total_correct += correct.sum().item()

        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'correct': correct[j].item(),
                'pred_object': 'NA' if pred_id[j].item() == -1 else dataset.ix_to_shape[pred_id[j].item()],
                'pred_confidence': pred_prob[j].item(),
                'target_object': dataset.ix_to_shape[target[j].item()]
            })

            if dataset.ix_to_shape[target[j].item()] not in obj_specific_results:
                obj_specific_results[dataset.ix_to_shape[target[j].item()]] = []

            obj_specific_results[dataset.ix_to_shape[target[j].item()]].append({
                'img_id': batch['img_ids'][j],
                'correct': correct[j].item(),
                'pred_object': 'NA' if pred_id[j].item() == -1 else dataset.ix_to_shape[pred_id[j].item()],
                'pred_confidence': pred_prob[j].item(),
                'target_object': dataset.ix_to_shape[target[j].item()]
            })

        acc = total_correct / total

        desc = f'Acc: {acc * 100:.2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('correct:', total_correct)
    print(f'Overall Acc: {acc * 100:.2f}%')

    for obj in obj_specific_results.keys():
        if len(obj_specific_results[obj]) == 0:
            continue
        obj_acc = sum([x['correct'] for x in obj_specific_results[obj]]) / len(obj_specific_results[obj])
        print(f'{obj} Acc: {obj_acc * 100:.2f}%')

    obj_specific_results['all'] = results

    return obj_specific_results


class ColorDataset(PaintSkillsDETREvaluationDataset):
    def __init__(self, image_dir, ann_path, metadata_path, args=None):
        super().__init__(image_dir, ann_path, metadata_path, args)

        self.color_to_ix = {color: i for i, color in enumerate(self.metadata['Color'])}
        self.ix_to_color = {i: color for color, i in self.color_to_ix.items()}

    def __getitem__(self, ix):
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)

        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)

        out['img'] = img
        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['target'] = []
        out['color_target'] = []
        out['target_names'] = []
        out['color_target_names'] = []
        out['img_ids'] = []
        out['imgs'] = []
        for datum in batch:
            target_names = []
            color_target_names = []


            shape = datum['objects'][0]['shape']
            if 'human' in shape:
                shape = 'human'
            # if not self.args.FT:
            shape = paintskills_object_to_coco_names(shape)

            shape_id = self.shape_to_ix[shape]
            out['target'].append(shape_id)
            out['img_ids'].append(datum['img_id'])

            color = datum['objects'][0]['color']
            color_id = self.color_to_ix[color]
            out['color_target'].append(color_id)

            target_names.append(shape)
            color_target_names.append(color)

            out['target_names'].append(target_names)
            out['color_target_names'].append(color_target_names)

            out['imgs'].append(datum['img'])

        out['target'] = torch.LongTensor(out['target'])
        out['color_target'] = torch.LongTensor(out['color_target'])

        return out

from matplotlib import pyplot as plt

def eval_color(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0
    total_object_correct = 0
    total_color_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        # [B, n_queries, num_classes]
        all_probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]

        # [B, num_classes]
        probas_max_query = all_probas.max(1).values

        # pred_id = probas.max(1).values.max(-1).indices

        # [B]
        pred_prob = probas_max_query.max(-1).values
        pred_id = probas_max_query.max(-1).indices

        # pred_id[pred_prob < 0.8] = -1

        # probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]  # [B, 100, n_class]
        # pred_id = probas.max(1).values.max(-1).indices  # [B, 100, n_class] -> [B, n_class] -> [B]
        target = batch["target"].to(device)
        object_correct = pred_id == target

        color_probas = outputs['pred_colors'].softmax(-1)[:, :, :-1]
        # pred_color_id = color_probas.max(1).values.max(-1).indices
        pred_obj_query_ids = all_probas.max(2).values.max(-1).indices

        # pred_color_id = color_probas.max(2).indices.gather(1, pred_obj_query_ids.view(1, B)).squeeze(0)

        color_on_queries = color_probas.max(2).indices
        pred_color_id = torch.stack([color_on_queries[b_i, query_id] for b_i, query_id in enumerate(pred_obj_query_ids.flatten())]).to(device)

        color_on_queries_prob = color_probas.max(2).values
        pred_color_prob = torch.stack([color_on_queries_prob[b_i, query_id]
                                      for b_i, query_id in enumerate(pred_obj_query_ids.flatten())]).to(device)

        color_target = batch["color_target"].to(device)
        color_correct = pred_color_id == color_target

        correct = object_correct * color_correct

        # break

        # correct = correct.float()

        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'target_names': batch['target_names'][j],
                'color_target_names': batch['color_target_names'][j],
                # 'pred_names': [dataset.ix_to_shape[x] for x in pred_id[j].view(-1).tolist()],
                'pred_object': 'NA' if pred_id[j].item() == -1 else dataset.ix_to_shape[pred_id[j].item()],
                'pred_color_names': [dataset.ix_to_color[x] for x in pred_color_id[j].view(-1).tolist()],
                'pred_object_confidence': pred_prob[j].item(),
                'pred_color_confidence': pred_color_prob[j].item(),
                'correct': correct[j].item(),
            })

        total += B
        total_correct += correct.sum().item()
        total_object_correct += object_correct.sum().item()
        total_color_correct += color_correct.sum().item()

        acc = total_correct / total
        object_acc = total_object_correct / total
        color_acc = total_color_correct / total

        desc = f'Acc: {acc * 100:.2f}% | Object Acc: {object_acc * 100:.2f}% | Color Acc: {color_acc * 100:.2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('# correct:', total_correct)
    print('# object correct:', total_object_correct)
    print('# color correct:', total_color_correct)
    print(f'Acc: {acc * 100:.2f}%')
    print(f'Object Acc: {object_acc * 100:.2f}%')
    print(f'Color Acc: {color_acc * 100:.2f}%')

    return results


class CountDataset(PaintSkillsDETREvaluationDataset):

    def __getitem__(self, ix):
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)


        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)

        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['target'] = []
        out['count_target'] = []
        out['img_ids'] = []

        for i, datum in enumerate(batch):
            out['img_ids'].append(datum['img_id'])

            for j, obj in enumerate(datum['objects']):
                shape = obj['shape']
                if 'human' in shape:
                    shape = 'human'
                # if not self.args.FT:
                shape = paintskills_object_to_coco_names(shape)
                shape_id = self.shape_to_ix[shape]

                if j == 0:
                    out['target'].append(shape_id)
                    out['count_target'].append(len(datum['objects']))
                else:
                    break

        out['target'] = torch.LongTensor(out['target'])
        out['count_target'] = torch.LongTensor(out['count_target'])

        return out


def eval_count(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0
    total_top_object_correct = 0
    total_count_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []
    obj_specific_results = {}
    count_specific_results = {}

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        pred_logits = outputs['pred_logits']

        # [B, num_queries, num_classes]
        probas = pred_logits.softmax(-1)[:, :, :-1]

        # [B, num_queries]
        max_prob = probas.max(-1).values

        assert max_prob.shape == (B, args.num_queries), max_prob.shape

        # [B]
        if args.p_threshold is not None:
            pred_n = (max_prob > args.p_threshold).sum(1)

        n_pred_objs_batch = pred_n.clamp(min=1)

        count_correct = pred_n == batch['count_target'].to(device)

        correct = torch.zeros(B).to(device)
        for i in range(B):
            gt_n = batch['count_target'][i].item()

            target_class_id = batch['target'][i].item()

            n_pred_objs = n_pred_objs_batch[i].item()

            # [num_queries]
            # top_n_indices = probas[i, :, target_class].topk(gt_n).indices
            top_n_indices = probas[i, :, target_class_id].topk(n_pred_objs).indices
            assert top_n_indices.shape == (n_pred_objs,), top_n_indices.shape

            # [n_pred_objs num_classes]
            probas_n_objs = probas[i][top_n_indices]
            assert probas_n_objs.shape == (n_pred_objs, args.num_classes)

            probas_n_objs_id = probas_n_objs.max(1).indices
            assert probas_n_objs_id.shape == (n_pred_objs,)

            # all predicted obj class should be GT obj classes
            correct_n = probas_n_objs_id == target_class_id
            correct[i] = correct_n.sum().item() == gt_n


        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'correct': correct[j].item(),
                'pred_count': n_pred_objs_batch[j].item(),
                'count_target': batch['count_target'][j].item(),
                'target_object': dataset.ix_to_shape[batch['target'][j].item()],
                'count_correct': count_correct[j].item(),
            })

            if dataset.ix_to_shape[batch['target'][j].item()] not in obj_specific_results:
                obj_specific_results[dataset.ix_to_shape[batch['target'][j].item()]] = []

            obj_specific_results[dataset.ix_to_shape[batch['target'][j].item()]].append({
                'img_id': batch['img_ids'][j],
                'correct': correct[j].item(),
                'pred_count': n_pred_objs_batch[j].item(),
                'count_target': batch['target'][j].item(),
                'count_correct': count_correct[j].item(),
                # 'target_object': dataset.ix_to_shape[batch['target'][j].item()]
            })

            if batch['count_target'][j].item() not in count_specific_results:
                count_specific_results[batch['count_target'][j].item()] = []

            count_specific_results[batch['count_target'][j].item()].append({
                'img_id': batch['img_ids'][j],
                'correct': correct[j].item(),
                'pred_count': n_pred_objs_batch[j].item(),
                'count_target': batch['count_target'][j].item(),
                'count_correct': count_correct[j].item(),
            })


        total += B
        total_correct += correct.sum().item()
        total_count_correct += count_correct.sum().item()

        acc = total_correct / total
        count_acc = total_count_correct / total

        desc = f'Acc: {acc * 100:.2f}% | count Acc: {count_acc * 100:.2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('# correct:', total_correct)
    print('# count correct:', total_count_correct)
    print(f'Acc: {acc * 100:.2f}%')
    print(f'count Acc: {count_acc * 100:.2f}%')

    for obj in obj_specific_results.keys():
        if len(obj_specific_results[obj]) == 0:
            continue
        obj_acc = sum([x['correct'] for x in obj_specific_results[obj]]) / len(obj_specific_results[obj])
        print(f'{obj} Acc: {obj_acc * 100:.2f}%')

    for count in count_specific_results.keys():
        if len(count_specific_results[count]) == 0:
            continue
        count_acc = sum([x['count_correct'] for x in count_specific_results[count]]) / len(count_specific_results[count])
        print(f'count-{count} Acc: {count_acc * 100:.2f}%')

    obj_specific_results['all'] = results
    obj_specific_results['count'] = count_specific_results

    return obj_specific_results


class SpatialDataset(PaintSkillsDETREvaluationDataset):

    def __getitem__(self, ix):
        datum = self.ann_data['data'][ix]

        out = deepcopy(datum)

        if self.args.gt_data_eval:
            fname = f"image_{datum['id']}"
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')
        else:
            fname = datum['id']
            img_path = self.image_dir.joinpath(fname).with_suffix('.png')

        img = Image.open(img_path)

        out['img_id'] = str(img_path)
        out['img'] = img

        out['width'] = img.width
        out['height'] = img.height

        img_tensor = self.img_transform(img)
        out['img_tensor'] = img_tensor

        return out

    def collate_fn(self, batch):
        out = deepcopy(batch[0])

        out['img_tensors'] = [x['img_tensor'] for x in batch]
        out['img_tensors'] = torch.stack(out['img_tensors'], 0)

        out['objA'] = []
        out['objB'] = []
        out['GT_objs'] = []
        out['relations_target'] = []

        out['img_ids'] = []
        out['imgs'] = []

        for i, datum in enumerate(batch):
            out['img_ids'].append(datum['img_id'])
            out['imgs'].append(datum['img'])

            for j, obj in enumerate(datum['objects']):
                shape = obj['shape']
                if 'human' in shape:
                    shape = 'human'

                # if not self.args.FT:
                shape = paintskills_object_to_coco_names(shape)

                shape_id = self.shape_to_ix[shape]

                out['GT_objs'].append(shape_id)

                if j == 0:
                    out['objA'].append(shape_id)
                    # out['count_target'].append(len(datum['objects']))
                elif j == 1:
                    out['objB'].append(shape_id)
                    relation = obj['relation']

                    assert relation is not None

                    # left_0
                    relation, relative_idx = relation.split('_')
                    assert relative_idx == '0'
                    assert relation in ['above', 'below', 'left', 'right']

                    out['relations_target'].append(relation)
                    # break

        out['objA'] = torch.LongTensor(out['objA'])
        out['objB'] = torch.LongTensor(out['objB'])
        out['GT_objs'] = torch.LongTensor(out['GT_objs'])

        return out


def eval_spatial(dataset, model, args):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    total = 0
    total_correct = 0
    total_obj_correct = 0
    # total_top_object_correct = 0
    total_spatial_correct = 0

    pbar = tqdm(total=len(dataloader))

    device = 'cuda'

    model = model.to(device)

    results = []
    # obj_specific_results = {}
    relation_specific_results = {}

    for batch in dataloader:
        inputs = batch['img_tensors']
        inputs = inputs.to(device)

        B = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)

        pred_logits = outputs['pred_logits']

        # [B, num_queries, num_classes]
        probas = pred_logits.softmax(-1)[:, :, :-1]

        # [B, num_queries]
        max_prob = probas.max(2).values
        class_id = probas.max(2).indices

        assert max_prob.shape == (B, args.num_queries), max_prob.shape

        # [B, 2]
        top2_query_indices = max_prob.topk(2, dim=1).indices.tolist()
        top2_query_probs = max_prob.topk(2, dim=1).values.tolist()

        assert len(batch['relations_target']) == B


        correct = torch.ones(B).to(device) * -1
        obj_correct = torch.ones(B).to(device) * -1

        pred_objs_ids = []
        gt_objs_ids = []
        pred_rels = []

        pred_obj_locs = []
        pred_obj_bbox = []

        for i in range(B):

            assert batch['relations_target'][i] in ['left', 'right', 'above', 'below']

            objA_class = batch['objA'][i].item()
            objB_class = batch['objB'][i].item()

            # pred_n_i = pred_n[i].item()

            # [num_queries]
            # top_n_indices = probas[i, :, target_class].topk(gt_n).indices
            # [2, num_classes]
            # top_n_indices = probas[i, :, :].topk(2, dim=0).indices

            assert len(top2_query_indices[i]) == 2, top2_query_indices[i]
            obj_C_query, obj_D_query = top2_query_indices[i]

            pred_obj_C = class_id[i, obj_C_query].item()
            pred_obj_D = class_id[i, obj_D_query].item()

            gt_objs_ids.append([objA_class, objB_class])
            pred_objs_ids.append([pred_obj_C, pred_obj_D])

            pred_obj_C_bbox = outputs['pred_boxes'][i, obj_C_query].tolist()
            pred_obj_D_bbox = outputs['pred_boxes'][i, obj_D_query].tolist()

            assert len(pred_obj_C_bbox) == 4, pred_obj_C_bbox

            # print('objA, objB', objA_class, objB_class)
            # print('objC, objD', pred_obj_C, pred_obj_D)

            # x_c, y_c, w, h = pred_obj_C_bbox
            x_C, y_C = pred_obj_C_bbox[:2]

            # x_c, y_c, w, h = pred_obj_D_bbox
            x_D, y_D = pred_obj_D_bbox[:2]

            x_diff = x_C - x_D
            y_diff = y_C - y_D

            pred_obj_bbox.append([pred_obj_C_bbox, pred_obj_D_bbox])
            pred_obj_locs.append([(x_C, y_C), (x_D, y_D)])

            if objA_class == objB_class:
                if objA_class == pred_obj_C and pred_obj_C == pred_obj_D:

                    # left/right
                    if abs(x_diff) > abs(y_diff):
                        if batch['relations_target'][i] in ['left', 'right']:
                            pred_rel = batch['relations_target'][i]
                            correct[i] = 1
                        else:
                            pred_rel = 'relation_incorrect'
                            correct[i] = 0
                    # above/below
                    else:
                        if batch['relations_target'][i] in ['above', 'below']:
                            pred_rel = batch['relations_target'][i]
                            correct[i] = 1
                        else:
                            pred_rel = 'relation_incorrect'
                            correct[i] = 0
                    obj_correct[i] = 1
                else:
                    pred_rel = 'obj_relation_incorrect'

                    correct[i] = 0
                    obj_correct[i] = 0

            else:
                if (objA_class, objB_class) == (pred_obj_C, pred_obj_D):
                    obj_correct[i] = 1

                    # left/right
                    if abs(x_diff) > abs(y_diff):
                        if x_C < x_D:
                            pred_rel = 'right'
                        else:
                            pred_rel = 'left'
                    # above/below
                    else:
                        if y_C > y_D:
                            pred_rel = 'above'
                        else:
                            pred_rel = 'below'

                    if pred_rel == batch['relations_target'][i]:
                        correct[i] = 1
                    else:
                        correct[i] = 0
                    # obj_correct[i] = 1

                    # import ipdb; ipdb.set_trace()

                elif (objA_class, objB_class) == (pred_obj_D, pred_obj_C):

                    obj_correct[i] = 1

                    # left/right
                    if abs(x_diff) > abs(y_diff):
                        if x_C < x_D:
                            pred_rel = 'left'
                        else:
                            pred_rel = 'right'

                    # above/below
                    else:
                        if y_C > y_D:
                            pred_rel = 'below'
                        else:
                            pred_rel = 'above'

                    if pred_rel == batch['relations_target'][i]:
                        correct[i] = 1
                    else:
                        correct[i] = 0
                    # obj_correct[i] = 1

                else:
                    pred_rel = 'obj_not_matching'
                    correct[i] = 0
                    obj_correct[i] = 0

                assert correct[i].min().item() in [0,1], correct[i]

            pred_rels.append(pred_rel)

        # # break
        # for j in range(B):
        #     results.append({
        #         'img_id': batch['img_ids'][j],
        #         'correct': correct[j].item(),
        #     })

        for j in range(B):
            results.append({
                'img_id': batch['img_ids'][j],
                'GT_objs': (dataset.ix_to_shape[gt_objs_ids[j][0]], dataset.ix_to_shape[gt_objs_ids[j][1]]),
                'pred_objs': (dataset.ix_to_shape[pred_objs_ids[j][0]], dataset.ix_to_shape[pred_objs_ids[j][1]]),
                'pred_obj_bbox': pred_obj_bbox[j],
                'pred_obj_locs': pred_obj_locs[j],
                'GT_rel': batch['relations_target'][j],
                'pred_rel': pred_rels[j],
                'correct': bool(correct[j].item()),
                'obj_correct': bool(obj_correct[j].item()),
            })


            if batch['relations_target'][j] not in relation_specific_results:
                relation_specific_results[batch['relations_target'][j]] = []

            relation_specific_results[batch['relations_target'][j]].append({
                'img_id': batch['img_ids'][j],
                'GT_objs': (dataset.ix_to_shape[gt_objs_ids[j][0]], dataset.ix_to_shape[gt_objs_ids[j][1]]),
                'pred_objs': (dataset.ix_to_shape[pred_objs_ids[j][0]], dataset.ix_to_shape[pred_objs_ids[j][1]]),
                'pred_obj_bbox': pred_obj_bbox[j],
                'pred_obj_locs': pred_obj_locs[j],
                'GT_rel': batch['relations_target'][j],
                'pred_rel': pred_rels[j],
                'correct': bool(correct[j].item()),
                'obj_correct': bool(obj_correct[j].item()),
            })

        total += B
        total_correct += correct.sum().item()
        total_obj_correct += obj_correct.sum().item()

        acc = total_correct / total
        obj_acc = total_obj_correct / total
        # color_acc = total_color_correct / total

        desc = f'Acc: {acc * 100: .2f} % | Obj Acc: {obj_acc * 100: .2f}%'

        pbar.set_description(desc)
        pbar.update(1)

    pbar.close()

    acc = total_correct / total

    print('Total:', total)
    print('# correct:', total_correct)
    # print('# count correct:', total_count_correct)
    print(f'Acc: {acc * 100:.2f}%')
    print(f'Obj Acc: {obj_acc * 100:.2f}%')


    for relation in relation_specific_results:
        if len(relation_specific_results[relation]) > 0:
            print(f'{relation}: {np.mean([r["correct"] for r in relation_specific_results[relation]]) * 100:.2f}%')

    relation_specific_results['all'] = results

    return relation_specific_results
    # return results
