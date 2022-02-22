# This code calibrates the three cameras with OpenCV and a pattern and saves the camera parameters into configuration files

#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRgb = pipeline.create(dai.node.XLinkOut)

LeftStr = "left"
RightStr = "right"
RgbStr = "rgb"
CfgStr = "cfg"

xoutLeft.setStreamName(LeftStr)
xoutRight.setStreamName(RightStr)
xoutRgb.setStreamName(RgbStr)

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Linking
camRgb.video.link(xoutRgb.input)
monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

# Connect to device
with dai.Device(pipeline) as device:

     # Output/input queues
    qLeft=device.getOutputQueue(LeftStr,4,False)
    qRight=device.getOutputQueue(RightStr,4,False)
    qRgb=device.getOutputQueue(RgbStr,4,False)

    while True:
        inLeft=qLeft.get()
        inRight=qRight.get()
        inRgb=qRgb.get()

        LeftFrame=inLeft.getFrame()
        RightFrame=inRight.getFrame()
        RgbFrame=inRgb.getCvFrame()

        cv2.imshow(LeftStr,LeftFrame)
        cv2.imshow(RightStr,RightFrame)
        cv2.imshow(RgbStr,RgbFrame)

        if cv2.waitKey(1) == ord('q'):
            break
