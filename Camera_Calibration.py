#!/usr/bin/python3
# Camera_Calibration.py calibrates the three cameras with OpenCV and a 8x6 chessboard pattern and 
# saves the camera parameters into configuration files
# Copyright (C) 2022  Simon Kaserer

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# This has to be executed in the Terminal since it has no GUI!

#!/usr/bin/env python3

import cv2
import depthai as dai
import os
import glob
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs of the OAK-D camera
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

        path_left='./CalPicsLeft/'
        path_right='./CalPicsRight/'
        path_Rgb='./CalPicsRGB/'
        path_data='./CalData/'

        # Check if the directories exist, if not create them:
        paths=[path_left,path_right,path_Rgb,path_data]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

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

            

            img_left=cv2.resize(LeftFrame,(400,300))
            img_right=cv2.resize(RightFrame,(400,300))
            img_Rgb=cv2.resize(RgbFrame,(400,300))
            #img_left=cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
            img_Rgb=cv2.cvtColor(img_Rgb,cv2.COLOR_BGR2GRAY)
            #img_right=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)            
            img_stacked=np.hstack((img_left,img_Rgb))
            img_stacked=np.hstack((img_stacked,img_right))
            cv2.putText(img_stacked,'Please take 12 Pictures by pressing the space bar to calibrate the cameras',(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)
            cv2.putText(img_stacked,'All of the three cameras should depict the pattern in its whole shape.',(0,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)
            cv2.imshow('Camera Calibration',img_stacked)

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
        objp = np.zeros((8*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
        objp_Rgb = np.zeros((8*6,3), np.float32)
        objp_Rgb[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        objp*=squaresize
        objp_Rgb*=squaresize

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints_left = [] # 2d points in image plane.
        imgpoints_right = []
        objpoints_Rgb = []
        imgpoints_Rgb = []

        images_left = glob.glob('./CalPicsLeft/*.jpg')
        images_right = glob.glob('./CalPicsRight/*.jpg')
        images_Rgb = glob.glob('./CalPicsRGB/*.jpg')
        images_left.sort()
        images_right.sort()
        images_Rgb.sort()
        if len(images_left)<10 :
            print('Not enough pictures found in CalPicsLeft!')
            quit()
        if len(images_right)<10 :
            print('Not enough pictures found in CalPicsRight!')
            quit()
        if len(images_Rgb)<10 :
            print('Not enough pictures found in CalPicsRGB!')
            quit()


        for imgLeft, imgRight in zip(images_left, images_right):

            imgL = cv2.imread(imgLeft)
            imgR = cv2.imread(imgRight)
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            retL, cornersL = cv2.findChessboardCorners(grayL, (8,6), None)
            retR, cornersR = cv2.findChessboardCorners(grayR, (8,6), None)

            # If found, add object points, image points (after refining them)
            if retL and retR == True:
                objpoints.append(objp)
                # Append the found corners to the imagepoint array
                cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                imgpoints_left.append(cornersL)
                cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
                imgpoints_right.append(cornersR)
        # Calibrate the left camera
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
        #Take an unused picture for the optimal camera matrix
        imgL=cv2.imread('./CalPicsLeft/CalLeft9.jpg')
        h,w=imgL.shape[:2]
        # Get the new camera matrix
        newcameramtx_left,roi_left=cv2.getOptimalNewCameraMatrix(mtx_left,dist_left,(w,h),1,(w,h))
        # save the matrices to .npy files
        np.save('./CalData/mtx_left.npy',mtx_left)
        np.save('./CalData/dist_left.npy',dist_left)
        np.save('./CalData/newcameramtx_left.npy',newcameramtx_left)
        print('Left camera calibrated!')
        # Calibrate the right camera
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)
        #Take an unused picture for the optimal camera matrix
        img=cv2.imread('./CalPicsRight/CalRight9.jpg')
        h,w=img.shape[:2]
        newcameramtx_right,roi_right=cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,(w,h),1,(w,h))
        # save the matrices to .npy files
        np.save('./CalData/mtx_right.npy',mtx_right)
        np.save('./CalData/dist_right.npy',dist_right)
        np.save('./CalData/newcameramtx_right.npy',newcameramtx_right)
        print('Right camera calibrated!')

        # Stereo Calibration
        flags=0
        # Fix the intrinsic camera matrices
        flags|=cv2.CALIB_FIX_INTRINSIC
        criteria_stereo=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
        ret_stereo,newCameraMtxL,distL,newCameraMtxR,distR,rot,trans,eMtx,fMtx=cv2.stereoCalibrate(objpoints,imgpoints_left,imgpoints_right,newcameramtx_left,dist_left,newcameramtx_right,dist_right,(w,h),criteria_stereo,flags)
        # Stereo Rectification
        rectL,rectR,projMtxL,projMtxR,Q,roi_L,roi_R=cv2.stereoRectify(newCameraMtxL,distL,newCameraMtxR,distR,(w,h),rot,trans,1,(0,0))
        stereoMapL=cv2.initUndistortRectifyMap(newCameraMtxL,distL,rectL,projMtxL,grayL.shape[::-1],cv2.CV_16SC2)
        stereoMapR=cv2.initUndistortRectifyMap(newCameraMtxR,distR,rectR,projMtxR,grayR.shape[::-1],cv2.CV_16SC2)
        #Saving Parameters
        np.save('./CalData/stereoMapL_x.npy',stereoMapL[0])
        np.save('./CalData/stereoMapL_y.npy',stereoMapL[1])
        np.save('./CalData/stereoMapR_x.npy',stereoMapR[0])
        np.save('./CalData/stereoMapR_y.npy',stereoMapR[1])
        print("Stereo Camera calibrated!")
        # Calibrate the Rgb camera
        for fname in images_Rgb[:11:]:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints_Rgb.append(objp_Rgb)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints_Rgb.append(corners)
            print(f'{fname} processed!')
        ret_Rgb, mtx_Rgb, dist_Rgb, rvecs_Rgb, tvecs_Rgb = cv2.calibrateCamera(objpoints_Rgb, imgpoints_Rgb, gray.shape[::-1], None, None)
        #Take an unused picture for the optimal camera matrix
        img=cv2.imread('./CalPicsRGB/CalRgb9.jpg')
        h,w=img.shape[:2]
        newcameramtx_Rgb,roi_Rgb=cv2.getOptimalNewCameraMatrix(mtx_Rgb,dist_Rgb,(w,h),1,(w,h))
        # save the matrices to .npy files
        np.save('./CalData/mtx_Rgb.npy',mtx_Rgb)
        np.save('./CalData/dist_Rgb.npy',dist_Rgb)
        np.save('./CalData/newcameramtx_Rgb.npy',newcameramtx_Rgb)

        print('Calibration process finished!')

        

if __name__=="__main__":
    main()
