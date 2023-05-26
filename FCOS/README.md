# FCOS: Fully Convolutional One-Stage Object Detection

## Installation
#### For a complete installation 
Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.


## Inference
The inference command line on voc split:

    python run_demo.py --weights "training_dir/fcos_imprv_R_50_FPN_1x/model_final.pth"

The checkpoint trained on voc can be download at <https://drive.google.com/file/d/1UEoA-0Sdj4K5HIFsnXWj4FH30Y34plWz/view?usp=drive_link>

## Training

The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x

Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
5) Now, training with 8 GPUs and 4 GPUs can have the same performance. Previous performance gap was because we did not synchronize `num_pos` between GPUs when computing loss. 
