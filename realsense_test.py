import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

ctx = rs.context()
devices = ctx.query_devices()
# serial_numbers = []
configs = []
pipelines = []
for device in devices:
    # serial_numbers.append(device.get_info(rs.camera_info.serial_number)) 
    pipelines.append(rs.pipeline(ctx))
    serial_number = device.get_info(rs.camera_info.serial_number)
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    configs.append(config)

# Start streaming
profiles = []
for i, config in enumerate(configs):
    profile = pipelines[i].start(config)
    profiles.append(profile)

path = "/home/ws1/instr/test_data"
try:
    while True:

        images = []
        for pipeline in pipelines:
          frames = pipeline.wait_for_frames()
          color_frame = frames.get_color_frame()
          if not color_frame:
              continue  
          color_image = np.asanyarray(color_frame.get_data())
          images.append(color_image)
        cv2.imwrite(path + "/left_rgb/left.png", images[0])
        cv2.imwrite(path + "/right_rgb/right.png", images[1])
        images = np.concatenate(images)
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)


finally:

    # Stop streaming
    pipeline.stop()