#!/usr/bin/env python3

# first, import all necessary modules
from pathlib import Path
import cv2
import depthai
import numpy as np

# bke
import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
bridge = CvBridge()
import time
import yaml
import os.path
from copy import deepcopy

# cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')

# see https://gist.github.com/rossbar/ebb282c3b73c41c1404123de6cea4771
def yaml_to_CameraInfo(yaml_fname):
    """
    Parse a yaml file containing camera calibration data (as produced by 
    rosrun camera_calibration cameracalibrator.py) into a 
    sensor_msgs/CameraInfo msg.
    
    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    with open(yaml_fname, "r") as file_handle:
        calib_data = yaml.safe_load(file_handle)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.K = calib_data["camera_matrix"]["data"]
    camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg

yaml_folder = os.path.dirname(__file__)
left_camera_info_msg = yaml_to_CameraInfo(yaml_folder+"/left.yaml")
right_camera_info_msg = yaml_to_CameraInfo(yaml_folder+"/right.yaml")
rgb_camera_info_msg = yaml_to_CameraInfo(yaml_folder+"/rgb.yaml")
depth_camera_info_msg = deepcopy(right_camera_info_msg)


rospy.init_node('image_converter', anonymous=True)
rgb_pub = rospy.Publisher("/cameras/rgb/image_raw/compressed",CompressedImage, queue_size = 1)
rgb_raw_pub = rospy.Publisher("/cameras/rgb/image_raw",Image, queue_size = 1)
rgb_info_pub = rospy.Publisher("/cameras/rgb/camera_info", CameraInfo, queue_size = 1)
left_info_pub = rospy.Publisher("/cameras/left/camera_info", CameraInfo, queue_size = 1)
right_info_pub = rospy.Publisher("/cameras/right/camera_info", CameraInfo, queue_size = 1)
left_pub = rospy.Publisher("/cameras/left/image_raw/compressed",CompressedImage, queue_size = 1)
left_raw_pub = rospy.Publisher("/cameras/left/image_raw",Image, queue_size = 1)
right_pub = rospy.Publisher("/cameras/right/image_raw/compressed",CompressedImage, queue_size = 1)
right_raw_pub = rospy.Publisher("/cameras/right/image_raw",Image, queue_size = 1)
disparity_raw_pub = rospy.Publisher("/cameras/disparity",Image, queue_size = 1)
disparity_pub = rospy.Publisher("/cameras/disparity/compressed",CompressedImage, queue_size = 1)
depth_pub = rospy.Publisher("/cameras/depth/image_raw", Image, queue_size = 1)
depth_compressed_pub = rospy.Publisher("/cameras/depth/image_raw/compressed",CompressedImage, queue_size = 1)
depth_info_pub = rospy.Publisher("/cameras/depth/camera_info", CameraInfo, queue_size = 1)
# /bke

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

fps = 4

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setFps(fps)
cam_rgb.setPreviewSize(640, 400) # For some reason, using native resolution cuts of sides, this provides wider field of view thatn 507x380
#cam_rgb.setPreviewSize(507, 380) # This strange number based on native image size of 4056x3040 / 8
cam_rgb.setInterleaved(False)


# see https://docs.luxonis.com/projects/api/en/latest/samples/02_mono_preview/#mono-preview
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
#cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_800_P) # native 1280x800
cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P) # native 1280x800
cam_left.setFps(fps) # max 120fps

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
#cam_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_800_P)  # native 1280x800
cam_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)  # native 1280x800, this is 640x400
cam_right.setFps(fps) # max 120fps

# see https://docs.luxonis.com/projects/api/en/latest/samples/03_depth_preview
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True # remove occlusions from left or right
cam_disparity = pipeline.createStereoDepth()
cam_disparity.setConfidenceThreshold(50) # 0-255, lower numbers are more restrictive
#cam_disparity.setOutputDepth(False)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
median = depthai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
cam_disparity.setMedianFilter(median)
cam_disparity.setDepthAlign(depthai.StereoDepthProperties.DepthAlign.RECTIFIED_RIGHT)
cam_disparity.setLeftRightCheck(lr_check)
max_disparity = 95

if extended_disparity: 
    max_disparity *= 2 # Double the range
    cam_disparity.setExtendedDisparity(extended_disparity)

if subpixel: 
    max_disparity *= 32 # 5 fractional bits, x32
    cam_disparity.setSubpixel(subpixel)

# When we get disparity to the host, we will multiply all values with the multiplier
# for better visualization
multiplier = 255 / max_disparity

cam_left.out.link(cam_disparity.left)
cam_right.out.link(cam_disparity.right)

# Next, we want a neural network that will produce the detections
detection_nn = pipeline.createNeuralNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
blob_path = str((Path(__file__).parent / Path('mobilenet-ssd/mobilenet-ssd.blob')).resolve().absolute())
print(f"blob path: {blob_path}")
detection_nn.setBlobPath(blob_path)
# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam_rgb.preview.link(detection_nn.input)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
control_in = pipeline.createXLinkIn()
control_in.setStreamName('control')
control_in.out.link(cam_rgb.inputControl)

xout_rgb = pipeline.createXLinkOut()
xout_left = pipeline.createXLinkOut()
xout_right = pipeline.createXLinkOut()
xout_disparity = pipeline.createXLinkOut()
xout_depth = pipeline.createXLinkOut()

# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
xout_left.setStreamName("left")
xout_right.setStreamName("right")
xout_disparity.setStreamName("disparity")
xout_depth.setStreamName("depth")

# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)
cam_left.out.link(xout_left.input)
cam_right.out.link(xout_right.input)
cam_disparity.disparity.link(xout_disparity.input)
cam_disparity.depth.link(xout_depth.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
device = depthai.Device(pipeline, True)
# And start. From this point, the Device will be in "running" mode and will start sending data via XLink
device.startPipeline()

# To consume the device results, we get two output queues from the device, with stream names we assigned earlier
q_rgb = device.getOutputQueue("rgb")
q_left = device.getOutputQueue("left")
q_right = device.getOutputQueue("right")
q_nn = device.getOutputQueue("nn")
q_disparity = device.getOutputQueue("disparity")
q_depth = device.getOutputQueue("depth")
q_control = device.getInputQueue("control")


# turn off autofocus

control_message = depthai.CameraControl()
control_message.setAutoFocusMode(depthai.CameraControl.AutoFocusMode.OFF)
#q_control.send(control_message)
lens_position = 125
#control_message = depthai.CameraControl()
control_message.setManualFocus(lens_position)
q_control.send(control_message)


# cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
#cv2.putText(frame, "lens_position: "+str(lens_position//10), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))


# Here, some of the default values are defined. Frame will be an image from "rgb" stream, bboxes will contain nn results

bboxes = []


# Since the bboxes returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
# receive the actual position of the bounding box on the image
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

lens_position = 0

# Main host-side application loop
while not rospy.is_shutdown():
    # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()
    in_left = q_left.tryGet()
    in_right = q_right.tryGet()
    in_disparity = q_disparity.tryGet()
    in_depth = q_depth.tryGet()

    timestamp = rospy.get_rostime()

    if in_rgb is None and in_nn is None and in_left is None  and in_right is None and in_disparity is None and in_depth is None:
        time.sleep(0.01)
        continue

    if in_rgb is not None and (rgb_pub.get_num_connections() or rgb_raw_pub.get_num_connections()):
        # rgb camera_info

        rgb_camera_info_msg.header.frame_id = "camera"
        rgb_camera_info_msg.header.stamp = timestamp
        rgb_info_pub.publish(rgb_camera_info_msg)


        # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        # Also, the array is transformed from CHW form into HWC
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        if True or image_pub.get_num_connections():
            frame = np.ascontiguousarray(frame)
            for raw_bbox in bboxes:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frame_norm(frame, raw_bbox)
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # frame = cv2.resize(frame,(200,200))
            if rgb_raw_pub.get_num_connections():
                msg = bridge.cv2_to_imgmsg(frame, "bgr8")
                msg.header.frame_id = "camera"
                msg.header.stamp = timestamp
                rgb_raw_pub.publish(msg)


            if rgb_pub.get_num_connections():
                msg = CompressedImage()
                msg.header.stamp = timestamp
                msg.header.frame_id = "camera"
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
                # Publish new image
                rgb_pub.publish(msg)

    if in_disparity is not None and (disparity_pub.get_num_connections() or disparity_raw_pub.get_num_connections()):
        frame_disparity = in_disparity.getFrame()
        frame_disparity = (frame_disparity*multiplier).astype(np.uint8)
        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        frame_disparity = cv2.applyColorMap(frame_disparity, cv2.COLORMAP_JET)
        if disparity_raw_pub.get_num_connections():
            msg = bridge.cv2_to_imgmsg(frame_disparity, "bgr8")
            msg.header.frame_id = 'stereo_right'
            msg.header.stamp = timestamp
            disparity_raw_pub.publish(msg)

        if disparity_pub.get_num_connections():
            frame_disparity = np.ascontiguousarray(frame_disparity)
            msg = CompressedImage()
            msg.header.stamp = timestamp
            msg.header.frame_id = "stereo_right"
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', frame_disparity)[1]).tobytes()
            # Publish new image
            disparity_pub.publish(msg)

    if in_depth is not None and (depth_pub.get_num_connections() or depth_compressed_pub.get_num_connections()):
        depth_camera_info_msg.header.frame_id = "stereo_right"
        depth_camera_info_msg.header.stamp = timestamp
        depth_info_pub.publish(depth_camera_info_msg)
        # Outputs ImgFrame message that carries RAW16 encoded (0..65535) depth data in millimeters.
        # Non-determined / invalid depth values are set to 0
        int_depth_frame = in_depth.getFrame()
        depth_frame = np.float32(int_depth_frame) / 1000.
        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        if depth_pub.get_num_connections():
            #msg = bridge.cv2_to_imgmsg(int_depth_frame, "mono16")
            msg = bridge.cv2_to_imgmsg(int_depth_frame, "passthrough")
            msg.header.frame_id = 'stereo_right'
            msg.header.stamp = timestamp
            depth_pub.publish(msg)
        
        if depth_compressed_pub.get_num_connections():
            msg = bridge.cv2_to_compressed_imgmsg(int_depth_frame, "png");
            msg.header.stamp = timestamp
            msg.header.frame_id = "camera"
            depth_compressed_pub.publish(msg)


    if in_left is not None and (left_pub.get_num_connections() or left_raw_pub.get_num_connections()):

        left_camera_info_msg.header.frame_id = "stereo_left"
        left_camera_info_msg.header.stamp = timestamp
        left_info_pub.publish(left_camera_info_msg)

        # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
        shape_left = (1, in_left.getHeight(), in_left.getWidth())
        # Also, the array is transformed from CHW form into HWC
        frame_left = in_left.getData().reshape(shape_left).transpose(1, 2, 0).astype(np.uint8)
        if left_raw_pub.get_num_connections():
            msg = bridge.cv2_to_imgmsg(frame_left, "mono8")
            msg.header.frame_id = 'stereo_left'
            msg.header.stamp = timestamp
            left_raw_pub.publish(msg)
        if left_pub.get_num_connections():
            frame_left = np.ascontiguousarray(frame_left)
            msg = CompressedImage()
            msg.header.frame_id = 'stereo_left'
            msg.header.stamp = timestamp
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', frame_left)[1]).tobytes()
            # Publish new image
            left_pub.publish(msg)



    if in_right is not None and (right_pub.get_num_connections() or right_raw_pub.get_num_connections()):

        right_camera_info_msg.header.frame_id = "stereo_right"
        right_camera_info_msg.header.stamp = timestamp
        right_info_pub.publish(right_camera_info_msg)

        # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
        shape_right = (1, in_right.getHeight(), in_right.getWidth())
        # Also, the array is transformed from CHW form into HWC
        frame_right = in_right.getData().reshape(shape_right).transpose(1, 2, 0).astype(np.uint8)
        if right_raw_pub.get_num_connections():
            msg = bridge.cv2_to_imgmsg(frame_right, "mono8")
            msg.header.frame_id = 'stereo_right'
            msg.header.stamp = timestamp
            right_raw_pub.publish(msg)
        if right_pub.get_num_connections():
            frame_right = np.ascontiguousarray(frame_right)
            msg = CompressedImage()
            msg.header.stamp = timestamp
            msg.header.frame_id = 'stereo_right'
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', frame_right)[1]).tobytes()
            right_pub.publish(msg)


    if in_nn is not None:
        # when data from nn is received, it is also represented as a 1D array initially, just like rgb frame
        bboxes = np.array(in_nn.getFirstLayerFp16())
        # the nn detections array is a fixed-size (and very long) array. The actual data from nn is available from the
        # beginning of an array, and is finished with -1 value, after which the array is filled with 0
        # We need to crop the array so that only the data from nn are left
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
        # next, the single NN results consists of 7 values: id, label, confidence, x_min, y_min, x_max, y_max
        # that's why we reshape the array from 1D into 2D array - where each row is a nn result with 7 columns
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        # Finally, we want only these results, which confidence (ranged <0..1>) is greater than 0.8, and we are only
        # interested in bounding boxes (so last 4 columns)
        bboxes = bboxes[bboxes[:, 2] > 0.8][:, 3:7]


