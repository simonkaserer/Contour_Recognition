#MCI Contour Extraction - Kaserer Simon 2022

#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import yaml
import Functions as fct
from contourGUI import *


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

    print("Contour Recognition Programme")
    
    # path='/home/pi/Desktop/Testcontours/'
    # res='4K_'
    # tool='Tixo'
    factor=0.0005
    thresh_val=150 #adoptable!
    counter=0
    every_nth_point=1
    flag=False
    connectpoints=False
    i=400

    # Load the calibration data
    mtx_Rgb=np.load('./CalData/mtx_Rgb.npy')
    dist_Rgb=np.load('./CalData/dist_Rgb.npy')
    newcameramtx_Rgb=np.load('./CalData/newcameramtx_Rgb.npy')

    app = QtWidgets.QApplication(sys.argv)
    ContourExtraction = QtWidgets.QMainWindow()
    gui = MainWindow(ContourExtraction,True)
    #gui.setupUi(ContourExtraction)
    ContourExtraction.show()
    #gui.Preview.setPixmap(QtGui.QPixmap('./test.jpg'))
    #image = cv2.imread('test.jpg')
    #update_contour(gui,image)
    

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
        edgeRgbFrame_undistorted=cv2.undistort(edgeRgbFrame,mtx_Rgb,dist_Rgb,None,newcameramtx_Rgb)
        cv2.imshow(edgeRgbStr, cv2.resize(edgeRgbFrame_undistorted,(700,500)))
        


        # add the contour extraction here:

        key = cv2.waitKey(50) 
        
        while (key!=ord('q')):
            # function call                     image ,DP-factor,TH value,pixel offset,use nth point,connect points,printisze,print number of points,show outer edge
            edge,img_edge=fct.extraction_polyDP(edgeRgbFrame_undistorted,factor,thresh_val,0,every_nth_point,connectpoints,False,False,False)
            
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
            if key==ord('7'):
                factor+=0.0001
                print(factor)
            if key==ord('8'):
                factor-=0.0001
                print(factor)
            if key==ord('0'):
                fct.dxf_exporter(edge,'/home/pi/Desktop/Testcontours/'+str(i)+'.dxf')
                i+=1
        
        

        

        key = cv2.waitKey(50)
        if flag is True:
            sys.exit()
        if key == ord('q'):
            sys.exit()
    
    sys.exit(app.exec_())
        
            


