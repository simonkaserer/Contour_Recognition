#MCI Contour Extraction - Kaserer Simon 2022

#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import Functions as fct

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
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
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
    
    path='/home/pi/Desktop'
    factor=0.0005
    thresh_val=170
    counter=0

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

        #edge=fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
        #print(edge)
        
        if fct.check_for_square(edgeRgbFrame,thresh_val):
            counter+=1
            if counter>3:
                #edge=fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
                edge,img_edge=fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
                cv2.imshow("tool-curve",img_edge)
        #print(hull)

        #
        
        
        #cv2.imshow("tool Contour",img_edge)

        

        key = cv2.waitKey(50)
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
            print('Saving contour to dxf...')
            fct.dxf_exporter(edge,path.append('toolcontour.dxf'),2)

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
        
            


