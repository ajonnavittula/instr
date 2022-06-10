import pyrealsense2 as rs
import numpy as np
import cv2
import sys
# path = "/home/ws1/instr/STIOS/zed/white_table/gt/"
# img = cv2.imread(path + "08.png")
# while True:
#     cv2.imshow('test', img)
#     cv2.waitKey(1)
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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    configs.append(config)

# Start streaming
profiles = []
for i, config in enumerate(configs):
    profile = pipelines[i].start(config)
    profiles.append(profile)

path = "/home/ws1/instr/test_data"
img_no = int(sys.argv[1])
try:
    while True:

        images = []
        for pipeline in pipelines:
          frames = pipeline.wait_for_frames()
          color_frame = frames.get_color_frame()
          depth_frame = frames.get_depth_frame()
          print(np.asanyarray(depth_frame.get_data()) * 1e-3)
          if not color_frame:
              continue  
          color_image = np.asanyarray(color_frame.get_data())
          images.append(color_image)

        # Show images
        images_show = np.concatenate(images)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images_show)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Quit")
            break
        elif k%256 == 32:
            cv2.imwrite(path + "/left_rgb/t_" + str(img_no) + ".png", images[0])
            cv2.imwrite(path + "/right_rgb/t_" + str(img_no) + ".png", images[1])
            print("saved img with id: t_" + str(img_no))
            img_no += 1

finally:

    # Stop streaming
    pipeline.stop()