# YOLOv3 in PyTorch

The repo implements the YOLOv3 in the PyTorch framework. Both inference and training modules are implemented.

## Introduction

You-Only-Look-Only (YOLO) newtorks was introduced by Joseph Redmon et al. 
Three versions were implemented in C, with the framework called [darknet](https://github.com/pjreddie/darknet) (paper: [v1](https://arxiv.org/abs/1506.02640), [v2](https://arxiv.org/abs/1612.08242), [v3](https://arxiv.org/abs/1804.02767)).

This repo implements the Nueral Network (NN) model of YOLOv3 in the PyTorch framework, aiming to ease the pain when the network needs to be modified or retrained.

There are a number of implementations existing in the open source domain, 
e.g., [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3),
[ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3),
[ultralytics/yolov3](https://github.com/ultralytics/yolov3), etc.
However, majority of them relies on "importing" the configuration file from the original darknet framework.
In this work, the model is built from scratch using PyTorch.

Additionally, both inference and training part are implemented. 
The original weights trained by the authors are converted to .pt file.
It can be used as a baseline for transfer learning.

**This project is licensed under BSD 3-Clause "Revised" License.**

## Getting Started

Before cloning the repo to your local machine, make sure that `git-lfs` is installed. See details about `git-lfs', see [this link](https://www.atlassian.com/git/tutorials/git-lfs#installing-git-lfs).

After `git-lfs` is installed. Run the following command to see sample detection results.
```
git lfs install
git clone https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch
cd YOLOv3-in-PyTorch
pip install -r requirements.txt
cd src
python main.py test --save-img

```
Detections will be saved in the `output` folder.

### Prerequisites

The repo is tested in `Python 3.7`. Additionally, the following packages are required: 

```
numpy
torch>=1.0
torchvision
pillow
matplotlib
```

## Structure

The repo is structured as following:
```
├── src
├── weights
│   ├── yolov3_original.pt
├── data
│   ├── coco.names
│   └── samples
├── requirements.txt
├── README.md
└── LICENSE
```

`src` folder contains the source code. 
`weights` folder contains the original weight file trained by Joseph Redmon et al.
`data/coco.names` file lists the names of the categories defined in the COCO dataset. 

## Usage

### Training

The weight trained by Joseph Redmon et al. is used as a starting point. 
The last few layers of the network can be unfreezed for transfer learning or finetuning. 

#### Training on COCO dataset

To train on COCO dataset, first you have to download the dataset from [COCO dataset website](http://cocodataset.org/#home).
Both images and the annotations are needed. 
Secondly, `pycocotools`, which serves as the Python API for COCO dataset needs to be installed.
Please follow the instructions on [their github repo](https://github.com/cocodataset/cocoapi) to install `pycocotools`.

After the COCO dataset is properly downloaded and the API setup, the training can be done by:

```
python main.py train --verbose --img-dir /path/to/image/folder --annot-path /path/to/annotation/file --reset-weights
```
You can see the network to converge within 1-2 epochs of training.

### Inference

To run inference on one image folder, run:

```
python main.py test --img-dir /path/to/image/folder --save-det --save-img
```

### Options

`main.py` provides numerous options to tweak the functions. Run `python main.py --help` to check the provided options.

## Authors

* **Haoyu Wu** - [wuhy08](https://github.com/wuhy08)

## License

This project is licensed under BSD 3-Clause "Revised" License - see the [LICENSE](LICENSE) file for details

## Credits

Original YOLOv3 Paper: [link](https://arxiv.org/pdf/1804.02767.pdf)

YOLOv3 C++ implementation: [link](https://github.com/pjreddie/darknet)
