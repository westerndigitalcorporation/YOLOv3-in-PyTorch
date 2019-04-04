# YOLOv3 in PyTorch

The repo implements the YOLOv3 in the PyTorch framework. Both inference and training modules are implemented.

## Introduction

TODO

## Getting Started

Run the following command to see sample detection results.
```
git clone http://...
cd [repo-dir]
pip install -r requirements.txt
cd src
python main.py test --save-img

```
Detections will be saved in the `output` folder.

### Prerequisites

The repo is tested in `Python 3.7+`. Additionally, the following packages are required: 

```
matplotlib
numpy
pillow
torch>=1.0
torchvision
```

## Structure

The repo is structured as following:
```
├── src
├── weights
├── checkpoints
├── config
├── data
│   ├── coco.names
│   └── samples
├── log
├── output
├── requirements.txt
├── README.md
└── LICENSE
```
## Usage

### Training

The weight trained by Joseph Redmon is used as a starting point. The last few layers of the network
can be unfreezed for transfer learning or finetuning. 

#### Training on COCO dataset

To train on COCO dataset, first you have to download the dataset from [COCO dataset website](http://cocodataset.org/#home).
Both images and the annotations are needed. Secondly, `pycocotools`, which serves as the Python API for COCO dataset needs to be installed.
Please follow the instructions on [their github repo](https://github.com/cocodataset/cocoapi) to install `pycocotools`.

After the COCO dataset is properly downloaded and the API setup, the training can be done by:

```
python main.py train --verbose --img-dir /path/to/images --annot-path /path/to/annotation/file --reset-weights
```
You can see the network to converge within 1-2 epochs of training.

### Inference

To run inference on one image folder, run:

```
python main.py test --img-dir /path/to/images --save-det --save-img
```

### Options

`main.py` provides numerous options to twick the functions. Run `python main.py --help` to check the provided options.

## Contributing

TODO

## Authors

* **Haoyu Wu** - [wuhy08](https://github.com/wuhy08)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under BSD 3-Clause "Revised" License - see the [LICENSE](LICENSE) file for details

## Credits

Original YOLOv3 Paper: [link](https://arxiv.org/pdf/1804.02767.pdf)

YOLOv3 C++ implementation: [link](https://github.com/pjreddie/darknet)