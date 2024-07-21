
# LayoutBench

The code for **LayoutBench**, a new diagnostic benchmark for layout-guided image generation, as described in the paper:

**[Diagnostic Benchmark and Iterative Inpainting for Layout-Guided Image Generation (CVPR 2024 Workshop)](https://layoutbench.github.io/)**

[Jaemin Cho](https://j-min.io),
[Linjie Li](https://www.microsoft.com/en-us/research/people/linjli/),
[Zhengyuan Yang](https://zyang-ur.github.io/),
[Zhe Gan](https://zhegan27.github.io/),
[Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/),
[Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

[[Project Page](https://layoutbench.github.io/)]
[[Paper](https://arxiv.org/abs/2304.06671)]



<img src="./assets/task_overview.png" width=1000px>


# Dataset

We host the dataset via Hugging Face Hub. Please download the dataset [https://huggingface.co/datasets/j-min/layoutbench](https://huggingface.co/datasets/j-min/layoutbench) and place it in the `datasets/` directory.

```bash
mkdir datasets
cd datasets
git clone https://huggingface.co/datasets/j-min/layoutbench
```

# Evaluation with pretrained DETR

## Download pretrained DETR checkpoint

```bash
wget https://huggingface.co/j-min/LayoutBench-DETR/resolve/main/checkpoint.pth
mkdir checkpoint/
mv checkpoint.pth checkpoint/layoutbench_detr_ckpt.pth
```

## Evaluation on LayoutBench splits

`SKILL_SPLIT` should be one of the following:
`number_few`, `number_many`, `position_boundary`, `position_center`, `size_tiny`, `size_large`, `shape_horizontal`, `shape_vertical`


```bash
LAYOUTBENCH_DIR='datasets/layoutbench'
SKILL_SPLIT='number_few'
DETR_CKPT_PATH='checkpoint/layoutbench_detr_ckpt.pth'
RUN_NAME='run' # name of the run
N_GPUS=4 # number of gpus
OUTPUT_DIR=output/$SKILL_SPLIT # directory to save the output

torchrun --nnodes=1 --nproc_per_node=$N_GPUS \
  detr/main.py \
  --batch_size 40 \
  --no_aux_loss \
  --eval \
  --backbone 'resnet101' \
  --dilation  \
  --dataset_file 'layoutbench_v1'  \
  --num_classes 48  \
  --resume $DETR_CKPT_PATH  \
  --layoutbench_dir $LAYOUTBENCH_DIR  \
  --skill_split $SKILL_SPLIT \
  --output_dir $OUTPUT_DIR \ 
  --save_viz # (optional) whether to save visualization
```



# Citation

If you find our project useful in your research, please cite the following paper:

```bibtex
@inproceedings{Cho2024LayoutBench,
  author    = {Jaemin Cho and Linjie Li and Zhengyuan Yang and Zhe Gan and Lijuan Wang and Mohit Bansal},
  title     = {Diagnostic Benchmark and Iterative Inpainting for Layout-Guided Image Generation},
  booktitle = {The First Workshop on the Evaluation of Generative Foundation Models},
  year      = {2024},
}
```
