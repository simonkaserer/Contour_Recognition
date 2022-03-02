# This code calibrates the three cameras with OpenCV and a pattern and saves the camera parameters into configuration files

#!/usr/bin/env python3

import cv2
import depthai as dai
import yaml
import glob
import numpy as np

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
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Linking
camRgb.video.link(xoutRgb.input)
monoLeft.out.link(xoutLeft.input)
monoRight.out.link(xoutRight.input)

squaresize=21 #in mm

def main():
    # Connect to device
    with dai.Device(pipeline) as device:

        # Output/input queues
        qLeft=device.getOutputQueue(LeftStr,4,False)
        qRight=device.getOutputQueue(RightStr,4,False)
        qRgb=device.getOutputQueue(RgbStr,4,False)

        path_left='/home/pi/MCI_Contour_Kaserer/MCI_Contour_Recognition/CalPicsLeft/'
        path_right='/home/pi/MCI_Contour_Kaserer/MCI_Contour_Recognition/CalPicsRight/'
        path_Rgb='/home/pi/MCI_Contour_Kaserer/MCI_Contour_Recognition/CalPicsRGB/'

        print('Calibration Program')
        print('Please take 12 Pictures by pressing the space bar to calibrate the cameras')
        print('All of the three cameras should depict the pattern in its whole shape and the whole picture should be filled')

        num_pic=1

        while num_pic<=12:
            inLeft=qLeft.get()
            inRight=qRight.get()
            inRgb=qRgb.get()

            LeftFrame=inLeft.getFrame()
            RightFrame=inRight.getFrame()
            RgbFrame=inRgb.getCvFrame()

            Left_small=LeftFrame
            Right_small=RightFrame
            Rgb_small=RgbFrame
            cv2.imshow(LeftStr,cv2.resize(Left_small,(400,300)))
            cv2.imshow(RightStr,cv2.resize(Right_small,(400,300)))
            cv2.imshow(RgbStr,cv2.resize(Rgb_small,(400,300)))

            str_img_left='CalLeft'+str(num_pic)
            str_img_right='CalRight'+str(num_pic)
            str_img_RGB='CalRgb'+str(num_pic)

            key=cv2.waitKey(10)
            if key == ord('q'):
                quit()
            if key ==ord(' '):
                cv2.imwrite((path_left+str_img_left+'.jpg'),LeftFrame)
                cv2.imwrite((path_right+str_img_right+'.jpg'),RightFrame)
                cv2.imwrite((path_Rgb+str_img_RGB)+'.jpg',RgbFrame)
                print(f"Saving picture number {num_pic}")
                num_pic+=1
        cv2.destroyAllWindows()
        # start the calibration process from the OpenCV documentation
        print('Start the calibration process')
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp_left = np.zeros((6*8,3), np.float32)
        objp_left[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
        objp_right = np.zeros((6*8,3), np.float32)
        objp_right[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
        objp_Rgb = np.zeros((6*8,3), np.float32)
        objp_Rgb[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        objp_left*=squaresize
        objp_right*=squaresize
        objp_Rgb*=squaresize

        # Arrays to store object points and image points from all the images.
        objpoints_left = [] # 3d point in real world space
        imgpoints_left = [] # 2d points in image plane.
        objpoints_right = []
        imgpoints_right = []
        objpoints_Rgb = []
        imgpoints_Rgb = []

        images_left = glob.glob('./CalPicsLeft/*.jpg')
        images_right = glob.glob('./CalPicsRight/*.jpg')
        images_Rgb = glob.glob('./CalPicsRGB*.jpg')

        for fname in images_left:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints_left.append(objp_left)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints_left.append(corners)
                # # Draw and display the corners
                # cv2.drawChessboardCorners(img, (7,6), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints_left, imgpoints_left, gray.shape[::-1], None, None)
        img=cv2.imread('./CalPicsLeft/CalLeft19.jpg')
        h,w=img.shape[:2]
        newcameramtx_left,roi_left=cv2.getOptimalNewCameraMatrix(mtx_left,dist_left,(w,h),1,(w,h))
        # save the matrices to .npy files
        np.save('./CalData/mtx_left.npy',mtx_left)
        np.save('./CalData/dist_left.npy',dist_left)
        np.save('./CalData/newcameramtx_left.npy',newcameramtx_left)

        for fname in images_right:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints_right.append(objp_right)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints_right.append(corners)
        
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints_right, imgpoints_right, gray.shape[::-1], None, None)
        img=cv2.imread('./CalPicsRight/CalRight19.jpg')
        h,w=img.shape[:2]
        newcameramtx_right,roi_right=cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,(w,h),1,(w,h))
        # save the matrices to .npy files
        np.save('./CalData/mtx_right.npy',mtx_right)
        np.save('./CalData/dist_right.npy',dist_right)
        np.save('./CalData/newcameramtx_right.npy',newcameramtx_right)

        for fname in images_Rgb:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints_Rgb.append(objp_Rgb)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints_Rgb.append(corners)

        ret_Rgb, mtx_Rgb, dist_Rgb, rvecs_Rgb, tvecs_Rgb = cv2.calibrateCamera(objpoints_Rgb, imgpoints_Rgb, gray.shape[::-1], None, None)
        img=cv2.imread('./CalPicsRGB/CalRgb19.jpg')
        h,w=img.shape[:2]
        newcameramtx_Rgb,roi_Rgb=cv2.getOptimalNewCameraMatrix(mtx_Rgb,dist_Rgb,(w,h),1,(w,h))
        # save the matrices to .npy files
        np.save('./CalData/mtx_Rgb.npy',mtx_Rgb)
        np.save('./CalData/dist_Rgb.npy',dist_Rgb)
        np.save('./CalData/newcameramtx_Rgb.npy',newcameramtx_Rgb)

        print('Calibration process finished!')

if __name__=="__main__":
    main()