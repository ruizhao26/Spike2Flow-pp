## [TPAMI 2026] Spike Camera Optical Flow Estimation Based on Continuous Spike Streams

<h4 align="center"> Rui Zhao<sup>1</sup>, Ruiqin Xiong<sup>1</sup>, Dongkai Wang<sup>2</sup>, Shiyu Xuan<sup>3</sup>, Jian Zhang<sup>4</sup>, Xiaopeng Fan<sup>5</sup>, Tiejun Huang<sup>1</sup> </h4>
<h4 align="center">1. State Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University<br>
2. School of Computing and Artificial Intelligence, Southwestern University of Finance and Economics<br>
3. School of Computer Science and Engineering, Nanjing University of Science and Technology<br>
4. School of Electronic and Computer Engineering, Peking University<br>
5. Department of Computer Science and Technology, Harbin Institute of Technology</h4><br> 

This repository contains the official source code for our paper:

Spike Camera Optical Flow Estimation Based on Continuous Spike Streams

TPAMI 2026

[Paper](https://ieeexplore.ieee.org/document/11316813)

## Environment

You can choose cudatoolkit version to match your server. The code is tested on PyTorch 2.0.1+cu117.

```bash
conda create -n spike2flowpp python==3.10.9
conda activate spike2flowpp
# You can choose the PyTorch version you like, we recommand version >= 1.10.1
# For example
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip3 install -r requirements.txt
```

## Prepare the Data

#### 1. Download and deploy the RSSF++ dataset

[Link of Real Scenes with Spike and Flow ++](https://github.com/ruizhao26/RSSF-pp)

#### 2. Set the path of RSSF-pp dataset in your serve

In the line2 of `configs/spike2flow.yml`

#### 3. Pre-processing for DSFT (Differential of Spike Firing Time)

It's not difficult to compute DSFT in real time, but in this version of the code, we choose to pre-processing the DSFT and save it in .h5 format to since the GPU memory resource is limited.

You can pre-processing the DSFT using the following command

```bash
cd datasets && 
python3 dat_to_DSFT_h5.py \
--rssf_root 'your root of RSSF dataset'\
--device 'cuda'
```

We will release the code of getting DSFT in real time in the future.

## Evaluate

```bash
cd shells
bash eval.py
```

## Train

```bash
cd shells
bash train.py
```

## Citations

If you find this code useful in your research, please consider citing our paper.

```
@article{zhao2026spike,
  title={Spike Camera Optical Flow Estimation Based on Continuous Spike Streams},
  author={Zhao, Rui and Xiong, Ruiqin and Wang, Dongkai and Xuan, Shiyu and Zhang, Jian and Fan, Xiaopeng and Huang, Tiejun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```

If you have any questions, please contact:  
ruizhao26@gmail.com