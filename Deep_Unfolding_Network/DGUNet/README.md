# Deep Generalized Unfolding Networks for Image Restoration         (CVPR 2022)
[Chong Mou](https://scholar.google.com.hk/citations?user=SYQoDk0AAAAJ&hl=en), Qian Wang, [Jian Zhang](https://jianzhang.tech/)

[Paper](https://arxiv.org/abs/2204.13348)

> **Abstract:** *Deep neural networks (DNN) have achieved great success in image restoration. However, most DNN methods are designed as a black box, lacking transparency and interpretability. Although some methods are proposed to combine traditional optimization algorithms with DNN, they usually demand pre-defined degradation processes or handcrafted assumptions, making it difficult to deal with complex and real-world applications. In this paper, we propose a Deep Generalized Unfolding Network (DGUNet) for image restoration. Concretely, without loss of interpretability, we integrate a gradient estimation strategy into the gradient descent step of the Proximal Gradient Descent (PGD) algorithm, driving it to deal with complex and real-world image degradation. In addition, we design inter-stage information pathways across proximal mapping in different PGD iterations to rectify the intrinsic information loss in most deep unfolding networks (DUN) through a multi-scale and spatial-adaptive way. By integrating the flexible gradient descent and informative proximal mapping, we unfold the iterative PGD algorithm into a trainable DNN. Extensive experiments on various image restoration tasks demonstrate the superiority of our method in terms of state-of-the-art performance, interpretability, and generalizability.* 

## :fire: Network Architecture
![Network](/figs/network.PNG)

## :art: Applications
### 🚩Deblurring🚩
 <img src="/figs/t_blur.PNG"   width="50%">
 
![blur](/figs/f_blur.PNG)

### 🚩Deraining🚩
![rain](/figs/t_rain.PNG)
![rain](/figs/f_rain.PNG)

### 🚩Denoising🚩
 <img src="/figs/t_noise.PNG"   width="50%">
 
 ![noise](/figs/f_noise.PNG)

### 🚩Compressive Sensing🚩
 ![noise](/figs/ft_cs.PNG)

## :wrench: Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).
The model is trained with 2 NVIDIA V100 GPUs.

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## :computer: Training and Evaluation

Training and Testing codes for deblurring, deraining, denoising and compressive sensing are provided in their respective directories.

## :european_castle: Model Zoo
### For Deblurring, Deraining, Denoising
Please download checkpoints from [Google Drive](https://drive.google.com/file/d/1bitvtmJAE1iKpFmdGx3OrN6Xti0JRPLc/view?usp=sharing).
### For Compressive Sensing
Please download checkpoints from [Google Drive](https://drive.google.com/file/d/1euV8SnXHuYswwbm9FaV1y5RQRLd58ou_/view?usp=sharing).

## 📑 Citation
If you use DGUNet, please consider citing:

    @inproceedings{Mou2022DGUNet,
        title={Deep Generalized Unfolding Networks for Image Restoration},
        author={Chong Mou and Qian Wang and Jian Zhang},
        booktitle={CVPR},
        year={2022}
    }

## :e-mail: Contact

If you have any question, please email `eechongm@gmail.com`.

## :hugs: Acknowledgements
This code is built on [MPRNet (PyTorch)](https://github.com/swz30/MPRNet). We thank the authors for sharing their codes of MPRNet.
