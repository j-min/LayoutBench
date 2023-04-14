# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, save_viz=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # if k is tensor: load to device, else: keep as is
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # For evaluation, this must be the original image size (before any data augmentation) <---------------
        # For visualization, this should be the image size after data augment, but before padding
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        ########### Added ##########

        # args = data_loader.dataset.args
        
        # Save bounding box overlay on image
        # if save_viz and batch_idx * args.batch_size <= 200:
        if save_viz:

            from util.viz_utils import plot_results, fig2img
            from PIL import Image
            from pathlib import Path
            from util.box_ops import box_cxcywh_to_xyxy

            blank_gt_bbox_dir = Path(output_dir) / "blank_gt_bbox_images"
            blank_gt_bbox_dir.mkdir(parents=True, exist_ok=True)

            gt_bbox_only_dir = Path(output_dir) / "gt_bbox_only_images"
            gt_bbox_only_dir.mkdir(parents=True, exist_ok=True)

            gen_gt_bbox_dir = Path(output_dir) / "gen_gt_bbox_images"
            gen_gt_bbox_dir.mkdir(parents=True, exist_ok=True)
            
            gen_gt_bbox_det_bbox_dir = Path(output_dir) / "gen_gt_bbox_det_bbox_images"
            gen_gt_bbox_det_bbox_dir.mkdir(parents=True, exist_ok=True)

            # viz_save_dir = Path(output_dir) / "bbox_images"
            # viz_save_dir = Path(output_dir) / "bbox_images"
            
            # viz_save_dir.mkdir(parents=True, exist_ok=True)
            # bbox_only_dir.mkdir(parents=True, exist_ok=True)

            # if args.clevr_resized_image_eval or args.layoutbench_resized_image_eval:

            #     # For evaluation, this must be the original image size (before any data augmentation)
            #     # For visualization, this should be the image size after data augment, but before padding <----
            viz_target_sizes = torch.LongTensor([[512, 512]] * len(targets)).to(device)
            viz_results = postprocessors['bbox'](outputs, viz_target_sizes)

            results = viz_results


            for i, target in enumerate(targets):
                image_id = target["image_id"].item()

                # img_path = data_loader.dataset.get_img_path(image_id)
                # img = Image.open(img_path).convert("RGB")

                img = target['transformed_img']
                img_path = target['img_path']

                # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

                # keep only predictions with 0.7+ confidence
                keep = results[i]['scores'] > 0.7
                keep = keep.tolist()

                # if args.clevr_resized_image_eval or args.layoutbench_resized_image_eval:

                #     # 512, 512
                #     W, H = img.size
                #     assert W == 512 and H == 512, f"Image size is {W}x{H}"
                #     det_normalized_xyxy_boxes = []
                #     # for j, box in enumerate(tgt['boxes'].tolist()):
                #     # for j, box in enumerate(results[i]['boxes'].tolist()):
                #     for j, box in enumerate(viz_results[i]['boxes'].tolist()):
                #         if keep[j]:
                #             # # box of 640x480 image
                #             x1, y1, x2, y2 = box
                #             det_normalized_xyxy_boxes.append([x1/W, y1/H, x2/W, y2/H])
                # else:
                W, H = img.size
                det_normalized_xyxy_boxes = []
                for j, box in enumerate(results[i]['boxes'].tolist()):
                    if keep[j]:
                        det_normalized_xyxy_boxes.append([box[0] / W, box[1] / H, box[2] / W, box[3] / H])

                det_box_captions = []
                for j, label in enumerate(results[i]['labels'].tolist()):
                    if keep[j]:
                        det_box_captions.append(data_loader.dataset.catid2name[label])


                gt_normzlied_xyxy_boxes = box_cxcywh_to_xyxy(target['boxes']).tolist()

                gt_box_captions = []
                for j, label in enumerate(target['labels'].tolist()):
                    gt_box_captions.append(data_loader.dataset.catid2name[label])

                # Save gt bbox only
                dummy_img = Image.new('RGB', (512, 512), (255, 255, 255))
                edge_color = 'blue'
                gt_box_colors = [edge_color] * len(gt_normzlied_xyxy_boxes)
                gt_bbox_only_img = fig2img(plot_results(
                    dummy_img,
                    gt_normzlied_xyxy_boxes,
                    gt_box_captions,
                    colors=gt_box_colors
                    ))
                gt_bbox_only_img_path = gt_bbox_only_dir / img_path.name
                if 'jpg' in img_path.name:
                    gt_bbox_only_img = gt_bbox_only_img.convert('RGB')
                gt_bbox_only_img.save(str(gt_bbox_only_img_path))
                # bbox_only_img_path = gt_bbox_only_dir / img_path.name
                # bbox_only_img.save(str(bbox_only_img_path))

                # Save generated image + gt_bbox
                edge_color = 'blue'
                gt_box_colors = [edge_color] * len(gt_normzlied_xyxy_boxes)
                gen_gt_bbox_img = fig2img(plot_results(
                    img,
                    gt_normzlied_xyxy_boxes,
                    gt_box_captions,
                    colors=gt_box_colors
                    ))
                gen_gt_bbox_img_path = gen_gt_bbox_dir / img_path.name
                if 'jpg' in img_path.name:
                    gen_gt_bbox_img = gen_gt_bbox_img.convert('RGB')
                gen_gt_bbox_img.save(str(gen_gt_bbox_img_path))

                # Save generated image + gt_bbox + det_bbox
                edge_color = 'red'
                det_box_colors = [edge_color] * len(det_normalized_xyxy_boxes)
                gen_gt_bbox_det_bbox_img = fig2img(plot_results(
                    img,
                    gt_normzlied_xyxy_boxes + det_normalized_xyxy_boxes,
                    gt_box_captions + det_box_captions,
                    colors=gt_box_colors + det_box_colors
                    ))
                gen_gt_bbox_det_bbox_img_path = gen_gt_bbox_det_bbox_dir / img_path.name
                if 'jpg' in img_path.name:
                    gen_gt_bbox_det_bbox_img = gen_gt_bbox_det_bbox_img.convert('RGB')
                gen_gt_bbox_det_bbox_img.save(str(gen_gt_bbox_det_bbox_img_path))                


        ############################



        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
