## [ICML 2024] Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow and Per-Block Quantization

This repository provides the official PyTorch implementation for [Jetfire: Efficient and Accurate Transformer Pretraining with
INT8 Data Flow and Per-Block Qua](https://arxiv.org/abs/2403.12422).

### News
- 2024.06: Our paper is selected as a spotlight paper!
- 2024.04: Jetfire is accepted by ICML 2024!

### Repository Overview
The Repocontains three main directories: `INT8_GPT2`, `Jetfire`, and `JetfireGEMMKernel`.

#### Directory Structure

```
Jetfire-INT8Training 
│
├── INT8_GPT2 # A INT8 training recipe
│   ├── train.py
│   ├── qmodel.py
│   └── ...
│
├── Jetfire # Implementation of linear and non-linear operators
│   ├── Linear
│   └── Nonlinear
│
└── JetfireGEMMKernel # CUDA Kernels of GEMM
    ├── setup.py
    ├── BlockQuantize
    └── ...
```

#### INT8_GPT2

The `INT8_GPT2` directory provides a recipe for INT8 training based on [nanoGPT](https://github.com/karpathy/nanoGPT.git). It includes necessary scripts and configurations to enable INT8 training for GPT-2 models. To use INT8 training, modify the `train.py` file as follows:

1. Open `train.py` and locate line 36.
2. Change the code to `use_quantize_model=True`.

This will enable the INT8 training mode.

The training command is
```sh
cd INT8_GPT2
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```
More details for training can be found in `INT8_GPT2/README.md`.

#### Jetfire

The `Jetfire` directory contains implementations for both linear and nonlinear operators. It is divided into two subdirectories: `Linear` and `Nonlinear`.

##### Linear operators

The `Linear` subdirectory contains implementations that utilize CUDA kernels from the `JetfireGEMMKernel` directory to perform linear operations in forward and backward process. The primary focus is on efficient matrix multiplications, which is introduced in Section 5 of [our paper](https://arxiv.org/pdf/2403.12422).

##### Nonlinear operators

The `Nonlinear` subdirectory contains implementations of nonlinear operators such as `GELU`, `LayerNorm`, `Quantize`, and `Stochastic Rounding`, leveraging Triton for optimal performance. This is introduced in Section 6 of [our paper](https://arxiv.org/pdf/2403.12422).

#### JetfireGEMMKernel

The `JetfireGEMMKernel` directory includes CUDA kernels specifically designed for matrix multiplication operations. These kernels are utilized by the `Linear` layer implementations in the `Jetfire` directory to achieve high-performance linear operations.

### Getting Started (Installation)

To get started with this repository, clone it and install the GEMM kernels:

```bash
git clone https://github.com/thu-ml/Jetfire-INT8Training.git
cd Jetfire-INT8Training

cd JetfireGEMMKernel
python setup.py install
cd ..
```

To install triton, we use this specific version because the API might change:
```bash
pip install https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/_packaging/07c94329-d4c3-4ad4-9e6b-f904a60032ec/pypi/download/triton-nightly/3.post20240610003544/triton_nightly-3.0.0.post20240610003544-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=ac2c36a49bf9c2bb780909b38096fb718f17efd78b88a1ca1d649f6d063cdc2c
```

For INT8 GPT-2 training, follow the instructions in the `INT8_GPT2` section above. For developing or experimenting with linear and nonlinear operators, please explore the `Jetfire` directories.

## Citation
If you find our work helpful or interesting, please cite our work :\)

```
@article{xi2024jetfire,
  title={Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow and Per-Block Quantization},
  author={Xi, Haocheng and Chen, Yuxiang and Zhao, Kang and Zheng, Kaijun and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2403.12422},
  year={2024}
}
```