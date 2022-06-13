"""
Collect images and depth information for new datasets using two Intel RealSense D415.
Depth information from the left camera is stored and used for training.
Ground truth Masks are generated using coco-annotator (https://github.com/jsbroks/coco-annotator) and create_masks.py.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs

class Camera:
    def __init__(self):
        self.configs = []
        self.serial_numbers = []
        self.pipelines = []
        self.profiles = []
        
        ctx = rs.context()
        devices = ctx.query_devices()
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)

            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

            pipeline = rs.pipeline(ctx)
            profile = pipeline.start(config)

            self.pipelines.append(pipeline)
            self.profiles.append(profile)
            self.serial_numbers.append(serial)
            self.configs.append(config)

        print("cameras initialized")


    def get_stereo(self):

        images = []
        for i,pipeline in enumerate(self.pipelines):
            frames = pipeline.wait_for_frames()
            # if i == 1:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            images.append(color_image)
        right = images[0]
        left = images[1]
        depth = np.asanyarray(depth_frame.get_data()) * 1e-3 #units: m instead of mm

        return left, right, depth

    def stop(self):
        for pipeline in self.pipelines:
            pipeline.stop()

def main():
    base_path = "/home/ws1/instr/SPS_dataset"
    img_no = int(sys.argv[1])

    cam = Camera()

    while True:
        left, right, depth = cam.get_stereo()

        # resize to avoid issues downstream
        left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (640, 480), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Apply colormap on depth image (change units back to mm)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth * 1e3, alpha=0.03), cv2.COLORMAP_JET)

        images = np.concatenate([left, right, depth_colormap], axis = 1)

        # show images
        cv2.namedWindow('Image Feed', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Image Feed', images)
        k = cv2.waitKey(1)

        # quit if "esc" pressed or save images if "space" is pressed
        if k%256 == 27:
            print("ESC pressed, exiting")
            cam.stop()
            break
        elif k%256 == 32:
            cv2.imwrite(base_path + "/left_rgb/" + str(img_no) + ".png", left)
            cv2.imwrite(base_path + "/right_rgb/" + str(img_no) + ".png", right)
            np.save(base_path + "/depth/" + str(img_no) + ".npy", depth)
            print("images saved as " + str(img_no))
            img_no += 1


if __name__ == '__main__':
    main()
