#MCI Contour Extraction - Kaserer Simon 2022

#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
# added
import os
from threading import Timer

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
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
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
    t = Timer(interval=7.0, function=saveimg)
    # start the timer
    t.start()
    factor=0.0005

    while(True):
        edgeLeft = edgeLeftQueue.get()
        edgeRight = edgeRightQueue.get()
        edgeRgb = edgeRgbQueue.get()

        edgeLeftFrame = edgeLeft.getFrame()
        edgeRightFrame = edgeRight.getFrame()
        edgeRgbFrame = edgeRgb.getFrame()
        
        

        # Show the frame
        cv2.imshow(edgeLeftStr, edgeLeftFrame)
        #cv2.imshow(edgeRightStr, edgeRightFrame)
        #cv2.imshow(edgeRgbStr, edgeRgbFrame)
       
        # add the contour extraction here:
        #gray=cv2.cvtColor(edgeLeftFrame,cv2.COLOR_BGR2GRAY)
        ret,thresh=cv2.threshold(edgeLeftFrame,200,255,0)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #find max area contour
        if len(contours) > 0:
            cnt = sorted(contours,key=cv2.contourArea)[-1]
            (x,y),(w,h),a=cv2.minAreaRect(cnt)
            print("Turning angle:")
            print(a)
            rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
            rotated_image=cv2.warpAffine(thresh,rot_mat,(int(w+x),int(h+y)))
        
            # Find the max-area contour of the outer line:
            cnts,hierarchy=cv2.findContours(rotated_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts)>0:
                cnt=sorted(cnts,key=cv2.contourArea)[-1]
                epsilon=0.1*cv2.arcLength(cnt,True)
                # use the Douglas-Peucker algorithm for approximating a rectangle shape
                approx=cv2.approxPolyDP(cnt,epsilon,True)
                #find a straight bounding rectangle inside the approximated one
                x,y,w,h=cv2.boundingRect(approx)
                
                # some pixel get cut away to reach the inner side of the contour
                offset=12
                # this offset can be smaller if the outer contour is straight and only 1 px 
                print("width")
                print(w-(2*offset))
                print("height")
                print(h-(2*offset))
                cropped_image=rotated_image[y+offset:y+h-offset,x+offset:x+w-offset]


                cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                #pic=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)

                if len(cnts)>0:
                    cnt=sorted(cnts,key=cv2.contourArea)[-1]
                    epsilon=0.0005*cv2.arcLength(cnt,True)
                    #   use the Douglas-Peucker algorithm for approximating a rectangle shape
                    approx=cv2.approxPolyDP(cnt,epsilon,True)
                    print("Number of appr. points:")
                    print(len(approx))
                    

                    #inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
                    cont_tool=cv2.drawContours(cropped_image,[approx],-1,(0,255,0),3)

                    cv2.imshow("tool-curve",cont_tool)

            
            
        

        key = cv2.waitKey(2000)
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
            factor=0.0005
        if key==ord('5'):
            factor=0.001
        if key==ord('6'):
            factor=0.0015
        if key==ord('7'):
            factor=0.0025
        if key==ord('8'):
            factor=0.005
        if key==ord('9'):
            factor=0.01
        if key==ord('0'):
            factor=0.02
            


