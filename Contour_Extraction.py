#MCI Contour Extraction - Kaserer Simon 2022

#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
# added
import os
import Functions as fct
#from threading import Timer

def saveimg():
    print ("Saving images!")
    cv2.imwrite(os.path.join(path,'testLeft.jpg'),edgeLeftFrame)
    cv2.imwrite(os.path.join(path,'testRight.jpg'),edgeRightFrame)
    cv2.imwrite(os.path.join(path,'testRgb.jpg'),edgeRgbFrame)



# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

edgeDetectorLeft = pipeline.create(dai.node.EdgeDetector)
edgeDetectorRight = pipeline.create(dai.node.EdgeDetector)
edgeDetectorRgb = pipeline.create(dai.node.EdgeDetector)

xoutEdgeLeft = pipeline.create(dai.node.XLinkOut)
xoutEdgeRight = pipeline.create(dai.node.XLinkOut)
xoutEdgeRgb = pipeline.create(dai.node.XLinkOut)
xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

edgeLeftStr = "edge left"
edgeRightStr = "edge right"
edgeRgbStr = "edge rgb"
edgeCfgStr = "edge cfg"

xoutEdgeLeft.setStreamName(edgeLeftStr)
xoutEdgeRight.setStreamName(edgeRightStr)
xoutEdgeRgb.setStreamName(edgeRgbStr)
xinEdgeCfg.setStreamName(edgeCfgStr)

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#camRgb.setVideoSize(1280,768)



monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
#monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
#monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)


edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())

# Linking
monoLeft.out.link(edgeDetectorLeft.inputImage)
monoRight.out.link(edgeDetectorRight.inputImage)
camRgb.video.link(edgeDetectorRgb.inputImage)

edgeDetectorLeft.outputImage.link(xoutEdgeLeft.input)
edgeDetectorRight.outputImage.link(xoutEdgeRight.input)
edgeDetectorRgb.outputImage.link(xoutEdgeRgb.input)

xinEdgeCfg.out.link(edgeDetectorLeft.inputConfig)
xinEdgeCfg.out.link(edgeDetectorRight.inputConfig)
xinEdgeCfg.out.link(edgeDetectorRgb.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output/input queues
    edgeLeftQueue = device.getOutputQueue(edgeLeftStr, 8, False)
    edgeRightQueue = device.getOutputQueue(edgeRightStr, 8, False)
    edgeRgbQueue = device.getOutputQueue(edgeRgbStr, 8, False)
    edgeCfgQueue = device.getInputQueue(edgeCfgStr)

    print("Switch between sobel filter kernels using keys '1' and '2'\nTo save the images press '3'")
    
    #added
    
    path='/home/pi/Desktop'
     #Set the timer to save a picture of each cam after 5 sec
    #t = Timer(interval=10.0, function=saveimg)
    # start the timer
    #t.start()
    factor=0.0005
    thresh_val=170

    while(True):
        edgeLeft = edgeLeftQueue.get()
        edgeRight = edgeRightQueue.get()
        edgeRgb = edgeRgbQueue.get()

        edgeLeftFrame = edgeLeft.getFrame()
        edgeRightFrame = edgeRight.getFrame()
        edgeRgbFrame = edgeRgb.getFrame()
        
        

        # Show the frame
        #cv2.imshow(edgeLeftStr, edgeLeftFrame)
        #cv2.imshow(edgeRightStr, edgeRightFrame)
        #cv2.imshow(edgeRgbStr, cv2.resize(edgeRgbFrame,(720,500)))
       
        # add the contour extraction here:
        fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,False,False)
        
            
            
        

        key = cv2.waitKey(1000)
        if key == ord('q'):
            break

        if key == ord('1'):
            print("Switching sobel filter kernel.")
            cfg = dai.EdgeDetectorConfig()
            sobelHorizontalKernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
            sobelVerticalKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
            cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            edgeCfgQueue.send(cfg)

        if key == ord('2'):
            print("Switching sobel filter kernel.")
            cfg = dai.EdgeDetectorConfig()
            sobelHorizontalKernel = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
            sobelVerticalKernel = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
            cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            edgeCfgQueue.send(cfg)
            
        #added
        if key==ord('3'):
            cv2.imwrite(os.path.join(path,'testLeftkey.jpg'),edgeLeftFrame)
            cv2.imwrite(os.path.join(path,'testRightkey.jpg'),edgeRightFrame)
            cv2.imwrite(os.path.join(path,'testRgbkey.jpg'),edgeRgbFrame)
            print('Saving images with key')

        if key==ord('4'):
            factor+=0.0001
            print("Factor:")
            print(factor)
        if key==ord('5') and factor>0.0001:
            factor-=0.0001
            print("Factor:")
            print(factor)
        if key==ord('6'):
            factor+=0.001
            print("Factor:")
            print(factor)
        if key==ord('7') and factor > 0.001:
            factor-=0.001
            print("Factor:")
            print(factor)
        if key==ord('8'):
            thresh_val+=10
            print("Threshold:")
            print(thresh_val)
        if key==ord('9') and thresh_val > 10:
            thresh_val-=10
            print("Threshold:")
            print(thresh_val)
        
            


