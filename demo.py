"""
Demo script.
"""

import os
import argparse
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from predictor import Predictor


class Camera:
    def __init__(self, *args, **kwargs):
        self.configs = []
        self.serial_numbers = []
        self.pipelines = []
        self.profiles = []
        self.img_path = "/home/ws1/instr/test_data"
        
        ctx = rs.context()
        devices = ctx.query_devices()
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)

            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

            pipeline = rs.pipeline(ctx)
            profile = pipeline.start(config)

            self.pipelines.append(pipeline)
            self.profiles.append(profile)
            self.serial_numbers.append(serial)
            self.configs.append(config)


        print("cameras initialized")
        # raise NotImplementedError('Please implement your camera class in "demo.py"')

    def get_stereo(self):
        # raise NotImplementedError('Please implement a method that returns a pair of stereo images (RGB, uint8 numpy arrays) in "demo.py"')
        images = []
        for pipeline in self.pipelines:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            images.append(color_image)
        left = images[0]
        right = images[1]
        # left = cv2.imread(self.img_path + "/left_rgb/22.png")
        # right = cv2.imread(self.img_path + "/right_rgb/22.png")
        return left, right


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth')
    # parser.add_argument('--focal-length', type=float, default=1390.0277099609375/(2208/640))  # ZED intrinsics per default
    parser.add_argument('--focal-length', type=float, default=1390.0277099609375/(1980/640))  # IntelRealsense D415 intrinsics per default
    parser.add_argument('--baseline', type=float, default=0.12)  # ZED intrinsics per default
    parser.add_argument('--viz', default=True, action='store_true')
    parser.add_argument('--save', default=True, action='store_true')
    parser.add_argument('--save-dir', type=str, default='./recorded_images')
    parser.add_argument('--aux-modality', type=str, default='depth', choices=['depth', 'disp'])
    parser.add_argument('--alpha', type=float, default=0.4)
    args = parser.parse_args()

    if args.save:
        print(f"Saving images to {args.save_dir}")
        os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'left'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'right'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'overlay'), exist_ok=True)

    # load net
    net = Predictor(state_dict_path=args.state_dict, focal_length=args.focal_length, baseline=args.baseline, return_depth=True if args.aux_modality == 'depth' else False)

    # init zed
    cam = Camera()

    ctr = 0
    # main forward loop
    while 1:
        left, right = cam.get_stereo()

        with torch.no_grad():
            pred_segmap, pred_depth = net.predict(left, right)

        if args.viz:
            left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imshow('left', cv2.resize(left.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
            cv2.imshow('right', cv2.resize(right.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
            cv2.imshow('pred', left_overlay)
            cv2.imshow(args.aux_modality, pred_depth / pred_depth.max())
            cv2.waitKey(1)

        if args.save:
            cv2.imwrite(os.path.join(args.save_dir, 'left', str(ctr).zfill(6) + '.png'), left)
            cv2.imwrite(os.path.join(args.save_dir, 'right', str(ctr).zfill(6) + '.png'), right)
            np.save(os.path.join(args.save_dir, 'depth', str(ctr).zfill(6) + '.npy'), pred_depth)
            cv2.imwrite(os.path.join(args.save_dir, 'segmap', str(ctr).zfill(6) + '.png'), pred_segmap)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imwrite(os.path.join(args.save_dir, 'overlay', str(ctr).zfill(6) + '.png'), left_overlay)

            ctr += 1


if __name__ == '__main__':
    demo()
