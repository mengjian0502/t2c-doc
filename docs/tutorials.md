# Tutorial

## Set up the Torch2Chip environment

To start, create the compatible environment based on the conda environment file `t2c.yml` :

```bash
conda env create -f t2c.yml
```

Specifically, the basic dependencies of torch2chip are:

```bash
python 3.9.19
torch==2.3.0
torchaudio==2.3.0
torchvision==0.18.0
huggingface-hub==0.23.1
timm==1.0.3
fxpmath==0.4.9
```

Activate your newly installed conda environment and start using Torch2Chip (see below)

```bash
# To activate this environment, use
#
#     $ conda activate t2c
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

## Compress your model

In Torch2Chip, we provide various example running scripts for different vision models, starting from the basic CNN (e.g., ResNet) to vision transformers (e.g., ViT). Given the needs of quickly compressing a pre-trained model, we employ post-training quantization (PTQ) as the major compression scheme for all the pre-trained FP32 model. 

List of examples are located here ([**link**](https://github.com/SeoLabCornell/torch2chip/tree/main/script/imagenet)).

Currently, Torch2Chip have provided all different kinds of combinations between quantization methods and pre-trained models. 

**Example:** Compressing a ViT-Small model on ImageNet with PTQ: 

```bash
if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vit_small
epochs=1
batch_size=100
lr=1e-4
loss=cross_entropy
weight_decay=1e-4
dataset="imagenet"
log_file="training.log"

wbit=8
abit=8
xqtype="lsq_token"
wqtype="minmax_channel"
num_samples=500
ttype=qattn
layer_train=True

save_path="./save/${dataset}/${model}/${xqtype}_${wqtype}/${model}_w${wbit}_a${abit}_lr${lr}_batch${batch_size}_${loss}loss_all/"

python3 -W ignore ./imagenet/vit.py \
    --save_path ${save_path} \
    --model ${model} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr ${lr} \
    --weight-decay ${weight_decay} \
    --batch_size ${batch_size} \
    --loss_type ${loss} \
    --dataset ${dataset} \
    --mixed_prec True \
    --optimizer adam \
    --trainer ${ttype} \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --num_samples ${num_samples} \
    --fine_tune \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --layer_trainer ${layer_train} \
```

**Arguments:**

`model`: Model architecture. 

- `vit_tiny`: `vit_tiny_patch16_224` pretrained by timm.

- `vit_small`: `vit_base_patch16_224` pretrained by timm.

- `vit_base`: `vit_base_patch16_224` pretrained by timm.

- `swin_tiny_patch4_window7_224`: `swin_tiny_patch4_window7_224` pretrained by timm.

- `swin_base_patch4_window7_224`: `swin_base_patch4_window7_224` pretrained by timm.

`xqtype`: Activation quantizer type

- `lsq`: Learned Step Size Quantization (LSQ) , Granularity: Tensor-wise.

`wqtype`: Weight quantizer type

- `minmax_channel`: Channel-wise MinMax Quantizer. 

`save_path`: Model saving path.

- By default, the compressed model will be saved inside the `./save/` folder under the torch2chip directory. 

`epochs`: Number of epochs for PTQ. 

- For fast compression process, we only run one time for each PTQ. 

`log_file`: Output log file that shows accuracy and model architecture. 

`lr`: Learning rate of PTQ (if there are trainable parameters in your quantizers).

`weight-decay`: Weight decay and regularization of PTQ (if there are trainable parameters in your quantizers).

`batch_size`: Batch size of PTQ calibration. 

`loss_type`: Calibration loss of PTQ. 

`dataset`: Dataset / task for the calibration. 

`mixed_prec`: Enable mixed precision training. 

`optimizer`: Optimizer of PTQ (if there are trainable parameters in your quantizers).

`trainer`: Calibration engine for post-training quantization (Details of different trainers can be found in the **Training** section. 

- `ptq`: Calibrator for CNN.
- `qattn`: Calibrator for vision transformers (inherited from `ptq`). 
- `smooth_quant`: Calibrator for SmoothQuant (inheirted from `qattn`).

`num_samples`: Total number of calibration samples (default = 500).

`fine_tune`: Flag for fine-tuning and reloading. 

`train_dir`: Directory of the training set of your dataset. 

`val_dir`: Directory of the validation set of your dataset. 

`layer_trainer`: Flag of controlling whether to optimize the trainable quantizer parameter (e.g., for LSQ) or just one shot calibration (e.g., for MinMax quantizer).

------

By executing the script above, torch2chip will start calibrating the model with compression. Please note that, if your model has been pruned by the pruner (link), torch2chip will collectively comrpess the sparse weights into the user-defined precision: \(W_Q = Quantize(W\times mask)\).

```bash
bash script/imagenet/vit-ptq-minmax-lsq.sh
```

After the calibration is completed, the fake-quantized model checkpoint is saved together with the log file:

```bash
cd ./save/imagenet/vit_small/lsq_minmax_channel/vit_small_w8_a8_lr1e-4_batch100_cross_entropyloss_all
```

## Convert the Fake-Quantized model to Torch2Chip model

Now you have a pre-trained fake-quantized model as a starting point. Now we can call T2C to fuse up the module and save all the intermediate input and output results. 

In the previous example, we use the learned step size quantization for activation and channel-wise minmax quantization for weights. To fuse the fake-quantized model, we elaborate the example of T2C into `t2c.py` with the execution script ` vit-t2c-minmax-lsq.sh`:

```bash
if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vit_small
epochs=200
batch_size=100
lr=0.1
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="t2c.log"
wbit=8
abit=8
xqtype="lsq_token"
wqtype="minmax_channel"
ttype=ptq

save_path="./save/imagenet/${model}/${xqtype}_${wqtype}/${model}_w${wbit}_a${abit}_lr1e-4_batch100_cross_entropyloss_all/t2c/"
pre_trained="./save/imagenet/${model}/${xqtype}_${wqtype}/${model}_w${wbit}_a${abit}_lr1e-4_batch100_cross_entropyloss_all/model_best.pth.tar"

python3 -W ignore ./imagenet/t2c.py \
    --save_path ${save_path} \
    --model ${model} \
    --batch_size ${batch_size} \
    --resume ${pre_trained} \
    --log_file ${log_file} \
    --fine_tune \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --evaluate \
    --trainer qattn \
    --swl 32 \
    --sfl 26 \
    --export_samples 1 \
```

**Arguments:**

Similar to the PTQ script, the post-compression fusion and model export follows the similar configuration except a couple of changes and unique argument variables.

- `save_path`: By default, the fused t2c model `t2c_model.pth.tar` and intermediate tensor files are saved in the `t2c/` folder under the path of PTQ directory. 
- `swl`: Total bitwidth of the fused scaling factor (BatchNorm + Quantization). Default = 32bit.
- `sfl`: Fractional bitwidth of the fused scaling factor (BatchNorm + Quantization). Default = 26bit.
- `export_samples`: Batch size of the final export tensors. Default = 1. 

By executing the bash file, torch2chip will load the pre-trained fake quantized model and run through the fusion and export process. The converted model and the extracted tensors are saved in the `save_path` and `save_path/tensors/`, respectively. 

```bash
bash script/imagenet/vit-t2c-minmax-lsq.sh
```

## Reload your Torch2Chip model back for verification

Now you have your T2C model (`t2c_model.pth.tar`) with integer-only operations. In practice, you might want to alter the low precision operations with some hardware-aware non-idealities. 

To reload the t2c model from the previous example (`lsq + minmax_channel`), we provide the following example (`vit-t2c-reload-minmax-lsq.sh`) as a starting point: 

```bash
if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

model=vit_small
batch_size=100
loss=cross_entropy
weight_decay=0.0005
dataset="imagenet"
log_file="reload.log"
wbit=8
abit=8
xqtype="lsq_token"
wqtype="minmax_channel"
ttype=ptq

save_path="./save/imagenet/vit_small/lsq_token_minmax_channel/vit_small_w8_a8_lr1e-4_batch100_cross_entropyloss_all/t2c/"
pre_trained="./save/imagenet/vit_small/lsq_token_minmax_channel/vit_small_w8_a8_lr1e-4_batch100_cross_entropyloss_all/t2c/t2c_model.pth.tar"

python3 -W ignore ./imagenet/reload.py \
    --save_path ${save_path} \
    --model ${model} \
    --batch_size ${batch_size} \
    --resume ${pre_trained} \
    --log_file ${log_file} \
    --fine_tune \
    --wqtype ${wqtype} \
    --xqtype ${xqtype} \
    --wbit ${wbit} \
    --abit ${abit} \
    --dataset ${dataset} \
    --train_dir "/share/seo/imagenet/train/" \
    --val_dir "/share/seo/imagenet/val/" \
    --evaluate \
    --trainer qattn \
```

As shown in the example of `imagenet/reload.py`, the model is reloaded to evaluate the accuracy based on the integer-only operation across the model. With that, users are allowed to customize the operator (e.g., Non-ideal MatMul) with hardware-oriented characteristics. 
