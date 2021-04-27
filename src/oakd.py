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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
bridge = CvBridge()
import time
# cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')

rospy.init_node('image_converter', anonymous=True)
image_pub = rospy.Publisher("/cameras/main",Image, queue_size = 1)
left_pub = rospy.Publisher("/cameras/left",Image, queue_size = 1)
right_pub = rospy.Publisher("/cameras/right",Image, queue_size = 1)
# /bke

# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setFps(15)
cam_rgb.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)


# see https://docs.luxonis.com/projects/api/en/latest/samples/02_mono_preview/#mono-preview
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P) # native 1280x800
cam_left.setFps(15) # max 120fps

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
cam_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)  # native 1280x800
cam_right.setFps(15) # max 120fps



# Next, we want a neural network that will produce the detections
detection_nn = pipeline.createNeuralNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
blob_path = str((Path(__file__).parent / Path('mobilenet-ssd/mobilenet-ssd.blob')).resolve().absolute())
print(f"blob path: {blob_path}")
detection_nn.setBlobPath(blob_path)
# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam_rgb.preview.link(detection_nn.input)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
xout_left = pipeline.createXLinkOut()
xout_right = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
xout_left.setStreamName("left")
xout_right.setStreamName("right")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)
cam_left.out.link(xout_left.input)
cam_right.out.link(xout_right.input)

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

# Here, some of the default values are defined. Frame will be an image from "rgb" stream, bboxes will contain nn results

bboxes = []


# Since the bboxes returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
# receive the actual position of the bounding box on the image
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


# Main host-side application loop
while not rospy.is_shutdown():
    # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()
    in_left = q_left.tryGet()
    in_right = q_right.tryGet()

    if in_rgb is None and in_nn and in_left is None:
        time.sleep(0.01)
        continue

    if in_rgb is not None:
        # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        # Also, the array is transformed from CHW form into HWC
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        if image_pub.get_num_connections():
            frame = np.ascontiguousarray(frame)
            for raw_bbox in bboxes:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frame_norm(frame, raw_bbox)
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # frame = cv2.resize(frame,(200,200))
            imgmsg = bridge.cv2_to_imgmsg(frame, "bgr8")
            imgmsg.header.frame_id = 'camera'
            image_pub.publish(imgmsg)

    if in_left is not None:
        # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
        shape_left = (1, in_left.getHeight(), in_left.getWidth())
        # Also, the array is transformed from CHW form into HWC
        frame_left = in_left.getData().reshape(shape_left).transpose(1, 2, 0).astype(np.uint8)
        if left_pub.get_num_connections():
            frame_left = np.ascontiguousarray(frame_left)
            left_msg = bridge.cv2_to_imgmsg(frame_left, "mono8")
            left_msg.header.frame_id = 'camera_left'
            left_pub.publish(left_msg)


    if in_right is not None:
        # When data from rgb stream is received, we need to transform it from 1D flat array into 3 x height x width one
        shape_right = (1, in_right.getHeight(), in_right.getWidth())
        # Also, the array is transformed from CHW form into HWC
        frame_right = in_right.getData().reshape(shape_right).transpose(1, 2, 0).astype(np.uint8)
        if right_pub.get_num_connections():
            frame_right = np.ascontiguousarray(frame_right)
            right_msg = bridge.cv2_to_imgmsg(frame_right, "mono8")
            right_msg.header.frame_id = 'camera_right'
            right_pub.publish(right_msg)


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


