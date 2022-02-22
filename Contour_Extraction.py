#MCI Contour Extraction - Kaserer Simon 2022

#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
# added
import os
#from threading import Timer

def saveimg():
    print ("Saving images!")
    cv2.imwrite(os.path.join(path,'testLeft.jpg'),edgeLeftFrame)
    cv2.imwrite(os.path.join(path,'testRight.jpg'),edgeRightFrame)
    cv2.imwrite(os.path.join(path,'testRgb.jpg'),edgeRgbFrame)


def extraction(img,printsize,printpoints):
    ret,thresh=cv2.threshold(img,thresh_val,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt = sorted(contours,key=cv2.contourArea)[-1]
        #Test: Show the found contour:
        pic=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        edge_out=cv2.drawContours(pic,[cnt],-1,(255,0,0),2)
        cv2.imshow("outer edge",cv2.resize(edge_out,(720,500)))
        # warp the contour into a straight rectangle:
            # find the cornerpoints of the square
        epsilon=0.01*cv2.arcLength(cnt,True)
            # use the Douglas-Peucker algorithm for approximating a rectangle shape
        outer_square=cv2.approxPolyDP(cnt,epsilon,True)
            # save the points in seperate variables
        if len(outer_square) ==4:
            pt_A=outer_square[0]
            pt_B=outer_square[1]
            pt_C=outer_square[2]
            pt_D=outer_square[3]
            #calculate the lengt and the width of the square
            width1=int(np.sqrt(((pt_A[0][0]-pt_D[0][0])**2)+(pt_A[0][1]-pt_D[0][1])**2))
            height1=int(np.sqrt(((pt_A[0][0]-pt_B[0][0])**2)+(pt_A[0][1]-pt_B[0][1])**2))
            width2=int(np.sqrt(((pt_B[0][0]-pt_C[0][0])**2)+(pt_B[0][1]-pt_C[0][1])**2))
            height2=int(np.sqrt(((pt_C[0][0]-pt_D[0][0])**2)+(pt_C[0][1]-pt_D[0][1])**2))
            width=max(width1,width2)
            height=max(height1,height2)
            if printsize:
                print("width, height")
                print(width,height)
            input_pts=np.float32([pt_A,pt_B,pt_C,pt_D])
            output_pts=np.float32([[0,0],[0,height],[width,height],[width,0]])
            transf_matrix=cv2.getPerspectiveTransform(input_pts,output_pts,)
            warped_image=cv2.warpPerspective(thresh,transf_matrix,(width,height),flags=cv2.INTER_LINEAR)
                # crop the image to remove the outer edge (offset can maybe be smaller when camera calibration is done?)
            offset=2
            warped_image=warped_image[0+offset:width-offset,0+offset:height-offset]

        # #resize the image to display properly
        # cv2.imshow("warpedtest",cv2.resize(warped_image,(600,600)))
        # key=cv2.waitKey(0)
        # if key==ord('q'):
        #     quit()
        



            cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            #cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)

        
            if len(cnts)>0:
                cnt=sorted(cnts,key=cv2.contourArea)[-1]
                epsilon=factor*cv2.arcLength(cnt,True)
                #   use the Douglas-Peucker algorithm for approximating a rectangle shape
                approx=cv2.approxPolyDP(cnt,epsilon,True)
                if printpoints:
                    print("Number of appr. points:")
                    print(len(approx))
                

                inv=cv2.cvtColor(warped_image,cv2.COLOR_GRAY2BGR)
                cont_tool=cv2.drawContours(inv,[approx],-1,(0,255,0),3)

                cv2.imshow("tool-curve",cv2.resize(cont_tool,(600,600)))
        else:
            print("Outer square not found!")


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
        extraction(edgeRgbFrame,False,False)
        
            
            
        

        key = cv2.waitKey(500)
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
        
            


