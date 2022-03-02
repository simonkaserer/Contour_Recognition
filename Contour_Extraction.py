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
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
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
    
    path='/home/pi/Desktop/Testcontours/'
    res='4K_'
    tool='Tixo'
    factor=0.0005
    thresh_val=150 #adoptable!
    counter=0
    every_nth_point=1
    flag=False
    connectpoints=False

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
        cv2.imshow(edgeRgbStr, cv2.resize(edgeRgbFrame,(720,500)))
       


        # add the contour extraction here:

        key = cv2.waitKey(50) 
        
        while (key!=ord('q')):
            edge,img_edge=fct.extraction_canny(edgeRgbFrame,factor,thresh_val,2,every_nth_point,connectpoints,False,True,True)
            #edge,img_edge=fct.extraction_convexHull(edgeRgbFrame,factor,thresh_val,2,every_nth_point,False,False,False)
            if img_edge is not None:
                cv2.imshow("tool Contour",img_edge)
            key = cv2.waitKey(50)
            if key == ord('q'):
                flag=True
                break
            if key == ord(' '):
                break
            if key == ord('1') and every_nth_point<30:
                every_nth_point+=1
                print(every_nth_point)
            if key == ord('2') and every_nth_point>1:
                every_nth_point-=1
                print(every_nth_point)
            if key==ord('3') and thresh_val<250:
                thresh_val+=10
                print(thresh_val)
            if key==ord('4') and thresh_val>30:
                thresh_val-=10
                print(thresh_val)
            if key==ord('5'):
                connectpoints=True
            if key==ord('6'):
                connectpoints=False

        
        

        

        key = cv2.waitKey(50)
        if flag is True:
            break
        if key == ord('q'):
            break

        if key == ord('1') and every_nth_point<20:
            #print("Switching sobel filter kernel.")
            #cfg = dai.EdgeDetectorConfig()
            #sobelHorizontalKernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
            #sobelVerticalKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
            #cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            #edgeCfgQueue.send(cfg)
            
            every_nth_point+=1
            print(every_nth_point)
            # print("save polyDP with every point")
            # fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # edge,img_edge=fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # cv2.imshow("tool Contour",img_edge)
            # cv2.waitKey(0)
            # fct.dxf_exporter(edge,path+'contourDP1'+tool+res+'.dxf',1)
            # fct.dxf_exporter_spline(edge,path+'splineDP1'+tool+res+'.dxf',1)
            

        if key == ord('2') and every_nth_point>1:
            #print("Switching sobel filter kernel.")
            #cfg = dai.EdgeDetectorConfig()
            #sobelHorizontalKernel = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
            #sobelVerticalKernel = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
            #cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            #edgeCfgQueue.send(cfg)
            every_nth_point-=1
            print(every_nth_point)
            # print("save polyDP with every 2nd point")
            # fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # edge,img_edge=fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # cv2.imshow("tool Contour",img_edge)
            # cv2.waitKey(0)
            # fct.dxf_exporter(edge,path+'contourDP2'+tool+res+'.dxf',2)
            # fct.dxf_exporter_spline(edge,path+'splineDP2'+tool+res+'.dxf',2)
            
            
        #added
        if key==ord('3') and thresh_val<250:
            #print('Saving contour to dxf...')
            thresh_val+=10
            print(thresh_val)
            # fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # edge,img_edge=fct.extraction_polyDP(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            # cv2.imshow("tool Contour",img_edge)
            # print("save polyDP with every 3rd point")
            # cv2.waitKey(0)
            # fct.dxf_exporter(edge,path+'contourDP3'+tool+res+'.dxf',3)
            # fct.dxf_exporter_spline(edge,path+'splineDP3'+tool+res+'.dxf',3)


        if key==ord('4') and thresh_val>30:
            #factor+=0.0001
            #print("Factor:")
            #print(factor)
            thresh_val-=10
            print(thresh_val)

            # fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            # fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            # edge,img_edge=fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            # print("save hull with every point")
            # cv2.imshow("tool Contour",img_edge)
            # cv2.waitKey(0)
            # fct.dxf_exporter(edge,path+'contourHull1'+tool+res+'.dxf',1)
            # fct.dxf_exporter_spline(edge,path+'splineHull1'+tool+res+'.dxf',1)
            
        if key==ord('5'):
            #factor-=0.0001
            #print("Factor:")
            #print(factor)
            fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            edge,img_edge=fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            print("save hull with every 2nd point")
            cv2.imshow("tool Contour",img_edge)
            cv2.waitKey(0)
            fct.dxf_exporter(edge,path+'contourHull2'+tool+res+'.dxf',2)
            fct.dxf_exporter_spline(edge,path+'splineHull2'+tool+res+'.dxf',2)
            
        if key==ord('6'):
            #factor+=0.001
            #print("Factor:")
            #print(factor)
            
            fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            edge,img_edge=fct.extraction_convexHull(edgeRgbFrame,thresh_val,0,False)
            print("save hull with every 3rd point")
            cv2.imshow("tool Contour",img_edge)
            cv2.waitKey(0)
            fct.dxf_exporter(edge,path+'contourHull3'+tool+res+'.dxf',3)
            fct.dxf_exporter_spline(edge,path+'splineHull3'+tool+res+'.dxf',3)
            
        if key==ord('7'):
            #factor-=0.001
            #print("Factor:")
            #print(factor)
            
            fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            edge,img_edge=fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            print("save TeH Chin with every point")
            cv2.imshow("tool Contour",img_edge)
            cv2.waitKey(0)
            fct.dxf_exporter(edge,path+'contourTC1'+tool+res+'.dxf',1)
            fct.dxf_exporter_spline(edge,path+'splineTC1'+tool+res+'.dxf',1)
            
        if key==ord('8'):
            #thresh_val+=10
            #print("Threshold:")
            #print(thresh_val)
            
            fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            edge,img_edge=fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            print("save TeH Chin with every 2nd point")
            cv2.imshow("tool Contour",img_edge)
            cv2.waitKey(0)
            fct.dxf_exporter(edge,path+'contourTC2'+tool+res+'.dxf',2)
            fct.dxf_exporter_spline(edge,path+'splineTC2'+tool+res+'.dxf',2)
            
        if key==ord('9'):
            #thresh_val-=10
            #print("Threshold:")
            #print(thresh_val)
            
            fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            edge,img_edge=fct.extraction_TehChin(edgeRgbFrame,factor,thresh_val,2,False,False,False)
            print("save TeH Chin with every 3rd point")
            cv2.imshow("tool Contour",img_edge)
            cv2.waitKey(0)
            fct.dxf_exporter(edge,path+'contourTC3'+tool+res+'.dxf',3)
            fct.dxf_exporter_spline(edge,path+'splineTC3'+tool+res+'.dxf',3)
        
            


