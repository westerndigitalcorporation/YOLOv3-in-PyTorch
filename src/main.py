# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import datetime
import json
import logging
import os
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.utils import collate_img_label_fn
from datasets.image import ImageFolder
from datasets.caltech import CaltechPedDataset
from datasets.coco import CocoDetectionBoundingBox
from inference import post_process
from model import YoloNetV3
from training import yolo_loss_fn
from utils import load_classes, untransform_bboxes, add_coco_empty_category, cxcywh_to_xywh, init_layer_randomly, draw_result


def parse_args():
    parser = argparse.ArgumentParser()
    # train or test:
    parser.add_argument('ACTION', type=str, help="'train' or 'test' the detector.")
    # data loading:
    # both:
    parser.add_argument('--dataset', dest='dataset_type', type=str, default='image_folder',
                        help="The type of the dataset used. Currently support 'coco', 'caltech' and 'image_folder'")
    parser.add_argument('--img-dir', dest='img_dir', type=str, default='../data/samples',
                        help="The path to the folder containing images to be detected or trained.")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4,
                        help="The number of sample in one batch during training or inference.")
    parser.add_argument('--n-cpu', dest='n_cpu', type=int, default=8,
                        help="The number of cpu thread to use during batch generation.")
    parser.add_argument("--img-size", dest='img_size', type=int, default=416,
                        help="The size of the image for training or inference.")
    # training only
    parser.add_argument('--annot-path', dest='annot_path', type=str, default=None,
                        help="TRAINING ONLY: The path to the file of the annotations for training.")
    parser.add_argument('--no-augment', dest='data_augment', action='store_false',
                        help="TRAINING ONLY: use this option to turn off the data augmentation of the dataset."
                             "Currently only COCO dataset support data augmentation.")

    # model loading:
    # both:
    parser.add_argument('--weight-path', dest='weight_path', type=str, default='../weights/yolov3_original.pt',
                        help="The path to weights file for inference or finetune training.")
    parser.add_argument('--cpu-only', dest='cpu_only', action='store_true',
                        help="Use CPU only no matter whether GPU is available.")
    parser.add_argument('--from-ckpt', dest='from_ckpt', action='store_true',
                        help="Load weights from checkpoint file, where optimizer state is included.")

    #training only:
    parser.add_argument('--reset-weights', dest='reset_weights', action='store_true',
                        help="TRAINING ONLY: Reset the weights which are not fixed during training.")
    parser.add_argument('--last-n-layers', dest='n_last_layers', type=str, default='1',
                        help="TRAINING ONLY: Unfreeze the last n layers for retraining.")

    # logging:
    # both:
    parser.add_argument('--log-dir', dest='log_dir', type=str, default='../log',
                        help="The path to the directory of the log files.")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Include INFO level log messages.")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Include DEBUG level log messages.")

    # saving:
    # inference only:
    parser.add_argument('--out-dir', dest='out_dir', type=str, default='../output',
                        help="INFERENCE ONLY: The path to the directory of output files.")
    parser.add_argument('--save-img', dest='save_img', action='store_true',
                        help="INFERENCE ONLY: Save output images with detections to output directory.")
    parser.add_argument('--save-det', dest='save_det', action='store_true',
                        help="INFERENCE ONLY: Save detection results in json format to output directory")
    # training only:
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, default="../checkpoints",
                        help="TRAINING ONLY: directory where model checkpoints are saved")
    parser.add_argument('--save-every-epoch', dest='save_every_epoch', type=int, default=1,
                        help="TRAINING ONLY: Save weights to checkpoint file every X epochs.")
    parser.add_argument('--save-every-batch', dest='save_every_batch', type=int, default=0,
                        help="TRAINING ONLY: Save weights to checkpoint file every X batches. "
                             "If value is 0, batch checkpoint will turn off.")

    # training parameters:
    parser.add_argument('--epochs', dest='n_epoch', type=int, default=30,
                        help="TRAINING ONLY: The number of training epochs.")
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=1E-4,
                        help="TRAINING ONLY: The training learning rate.")

    # inference parameters:
    parser.add_argument('--class-path', dest='class_path', type=str, default='../data/coco.names',
                        help="TINFERENCE ONLY: he path to the file storing class label names.")
    parser.add_argument('--conf-thres', dest='conf_thres', type=float, default=0.8,
                        help="INFERENCE ONLY: object detection confidence threshold during inference.")
    parser.add_argument('--nms-thres', dest='nms_thres', type=float, default=0.4,
                        help="INFERENCE ONLY: iou threshold for non-maximum suppression during inference.")
    _options = parser.parse_args()
    return _options


def config_logging(log_dir, log_file_name, level=logging.WARNING, screen=True):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file_name)
    _handlers = [logging.FileHandler(log_path)]
    if screen:
        _handlers.append(logging.StreamHandler())
    logging.basicConfig(level=level, handlers=_handlers)


def config_device(cpu_only: bool):
    if not cpu_only:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logging.warning('CUDA device is not available. Will use CPU')
    else:
        use_cuda = False
    _device = torch.device("cuda:0" if use_cuda else "cpu")
    return _device


def load_yolov3_model(weight_path, device, ckpt=False, mode='eval'):
    _model = YoloNetV3(nms=True)
    if not ckpt:
        _model.load_state_dict(torch.load(weight_path))
    else:
        _model.load_state_dict(torch.load(weight_path)['model_state_dict'])
    _model.to(device)
    if mode == 'eval':
        _model.eval()
    elif mode == 'train':
        _model.train()
    else:
        raise ValueError("YoloV3 model can be only loaded in 'train' or 'eval' mode.")
    return _model


def load_dataset(type, img_dir, annot_dir, img_size, batch_size, n_cpu, shuffle, augment, **kwargs):
    if type == "image_folder":
        _dataset = ImageFolder(img_dir, img_size=img_size)
        _collate_fn = None
    elif type == "coco":
        _transform = 'random' if augment else 'default'
        _dataset = CocoDetectionBoundingBox(img_dir, annot_dir, img_size=img_size, transform=_transform)
        _collate_fn = collate_img_label_fn
    elif type == "caltech":
        _dataset = CaltechPedDataset(img_dir, img_size, **kwargs)
        _collate_fn = collate_img_label_fn
    else:
        raise TypeError("dataset types can only be 'image_folder', 'coco' or 'caltech'.")
    if _collate_fn is not None:
        _dataloader = DataLoader(_dataset, batch_size, shuffle, num_workers=n_cpu, collate_fn=_collate_fn)
    else:
        _dataloader = DataLoader(_dataset, batch_size, shuffle, num_workers=n_cpu)
    return _dataloader


def make_output_dir(out_dir):
    if os.path.exists(out_dir):
        logging.warning(
            'The output folder {} exists. New output may overwrite the old output.'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    return


def run_detection(model, dataloader, device, conf_thres, nms_thres):
    results = []
    _detection_time_list = []
    # _total_time = 0

    logging.info('Performing object detection:')

    for batch_i, batch in enumerate(dataloader):
        file_names = batch[0]
        img_batch = batch[1].to(device)
        scales = batch[2].to(device)
        paddings = batch[3].to(device)

        # Get detections
        start_time = time.time()
        with torch.no_grad():
            detections = model(img_batch)
        detections = post_process(detections, True, conf_thres, nms_thres)

        for detection, scale, padding in zip(detections, scales, paddings):
            detection[..., :4] = untransform_bboxes(detection[..., :4], scale, padding)
            cxcywh_to_xywh(detection)

        # Log progress
        end_time = time.time()
        inference_time_both = end_time - start_time
        # print("Total PP time: {:.1f}".format(inference_time_pp*1000))
        logging.info('Batch {}, '
                     'Total time: {}s, '.format(batch_i,
                                                inference_time_both))
        _detection_time_list.append(inference_time_both)
        # _total_time += inference_time_both

        results.extend(zip(file_names, detections, scales, paddings))

    _detection_time_tensor = torch.tensor(_detection_time_list)
    avg_time = torch.mean(_detection_time_tensor)
    time_std_dev = torch.std(_detection_time_tensor)
    logging.info('Average inference time (total) is {}s.'.format(float(avg_time)))
    logging.info('Std dev of inference time (total) is {}s.'.format(float(time_std_dev)))
    return results


def run_training(model, optimizer, dataloader, device, img_size, n_epoch, every_n_batch, every_n_epoch, ckpt_dir):
    losses = None
    for epoch_i in range(n_epoch):
        for batch_i, (imgs, targets, target_lengths) in enumerate(dataloader):
            with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                imgs = imgs.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                result = model(imgs)
                try:
                    losses = yolo_loss_fn(result, targets, target_lengths, img_size, False)
                    losses[0].backward()
                except RuntimeError as e:
                    logging.error(e)
                    optimizer.zero_grad()
                    continue
                optimizer.step()

            logging.info(
                "[Epoch {}/{}, Batch {}/{}] [Losses: total {}, coord {}, obj {}, noobj {}, class {}]"
                .format(
                    epoch_i,
                    n_epoch,
                    batch_i,
                    len(dataloader),
                    losses[0].item(),
                    losses[1].item(),
                    losses[2].item(),
                    losses[3].item(),
                    losses[4].item()
                )
            )

            # logging.info(
            #     "{}, {}, {}, {}, {}, {}, {}"
            #     .format(
            #         epoch_i,
            #         batch_i,
            #         losses[0].item(),
            #         losses[1].item(),
            #         losses[2].item(),
            #         losses[3].item(),
            #         losses[4].item()
            #     )
            # )

            if every_n_batch != 0 and (batch_i + 1) % every_n_batch == 0:
                save_path = "{}/ckpt_epoch_{}_batch_{}.pt".format(ckpt_dir, epoch_i, batch_i)
                save_checkpoint_weight_file(model, optimizer, epoch_i, batch_i, losses, save_path)

        if (epoch_i + 1) % every_n_epoch == 0:
            save_path = "{}/ckpt_epoch_{}.pt".format(ckpt_dir, epoch_i)
            save_checkpoint_weight_file(model, optimizer, epoch_i, 0, losses, save_path)

    return


def save_results_as_json(results, json_path):
    results_json = []
    for result_raw in results:
        path, detections, _, _ = result_raw
        image_id = os.path.basename(path)
        image_id, _ = os.path.splitext(image_id)
        try:
            image_id = int(image_id)
        except ValueError:
            pass
        for detection in detections:
            detection = detection.tolist()
            bbox = detection[:4]
            score = detection[4]
            category_id = add_coco_empty_category(int(detection[5]))
            result = {'image_id': image_id, 'category_id': category_id, 'bbox': bbox, 'score': score}
            results_json.append(result)
    with open(json_path, 'w') as f:
        json.dump(results_json, f)
    return


def save_det_image(img_path, detections, output_img_path, class_names):
    img = Image.open(img_path)
    # Draw bounding boxes and labels of detections
    if detections is not None:
        img = draw_result(img, detections, class_names=class_names)
    img.save(output_img_path)
    return


def save_results_as_images(results, output_dir, class_names):
    logging.info('Saving images:')
    # Iterate through images and save plot of detections
    for img_i, result in enumerate(results):
        path, detections, _, _ = result
        logging.info("({}) Image: '{}'".format(img_i, path))
        # Create plot
        img_output_filename = '{}/{}.png'.format(output_dir, img_i)
        save_det_image(path, detections, img_output_filename, class_names)
    return


def save_checkpoint_weight_file(model, optimizer, epoch, batch, loss, weight_file_path):
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, weight_file_path)
    logging.info("saving model at epoch {}, batch {} to {}".format(epoch, batch, weight_file_path))
    return


def run_yolo_inference(opt):
    # configure logging
    current_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name_by_time = current_datetime_str + ".log"
    if options.debug:
        log_level = logging.DEBUG
    elif options.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    config_logging(opt.log_dir, log_file_name_by_time, level=log_level)
    # set the device for inference
    dev = config_device(opt.cpu_only)
    make_output_dir(opt.out_dir)
    # load model
    model = load_yolov3_model(opt.weight_path, dev, ckpt=opt.from_ckpt)
    # load data
    dataloader = load_dataset(type='image_folder',
                              img_dir=opt.img_dir,
                              annot_dir=None,
                              img_size=opt.img_size,
                              batch_size=opt.batch_size,
                              n_cpu=opt.n_cpu,
                              shuffle=False,
                              augment=False)
    # run detection
    results = run_detection(model, dataloader, dev, opt.conf_thres, opt.nms_thres)
    # post processing
    if opt.save_det:
        json_path = '{}/{}/detections.json'.format(opt.out_dir, current_datetime_str)
        make_output_dir(os.path.split(json_path)[0])
        save_results_as_json(results, json_path)
    if opt.save_img:
        class_names = load_classes(opt.class_path)
        img_path = '{}/{}/img'.format(opt.out_dir, current_datetime_str)
        make_output_dir(img_path)
        save_results_as_images(results, img_path, class_names)
    return


def run_yolo_training(opt):
    # configure logging
    current_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name_by_time = current_datetime_str + ".log"
    if opt.debug:
        log_level = logging.DEBUG
    elif opt.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    config_logging(opt.log_dir, log_file_name_by_time, level=log_level)
    # configure device
    dev = config_device(opt.cpu_only)
    ckpt_dir = '{}/{}'.format(opt.ckpt_dir, current_datetime_str)
    os.makedirs(ckpt_dir, exist_ok=True)
    model = load_yolov3_model(opt.weight_path, dev, ckpt=opt.from_ckpt, mode='train')
    finetune_layers = model.yolo_last_n_layers(opt.n_last_layers)

    for p in model.parameters():
        p.requires_grad = False
    for layer in finetune_layers:
        if opt.reset_weights:
            layer.apply(init_layer_randomly)
        for p in layer.parameters():
            p.requires_grad_()

    dataloader = load_dataset(type=opt.dataset_type,
                              img_dir=opt.img_dir,
                              annot_dir=opt.annot_path,
                              img_size=opt.img_size,
                              batch_size=opt.batch_size,
                              n_cpu=opt.n_cpu,
                              shuffle=True,
                              augment=opt.data_augment)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.learning_rate
    )

    run_training(model, optimizer, dataloader, dev,
                 opt.img_size,
                 opt.n_epoch,
                 opt.save_every_batch,
                 opt.save_every_epoch,
                 ckpt_dir)
    return


if __name__ == '__main__':
    options = parse_args()
    if options.ACTION == 'train':
        run_yolo_training(options)
    elif options.ACTION == 'test':
        run_yolo_inference(options)
    else:
        raise ValueError("Only action of 'train' or 'test' supported.")



