# This file is for testing code snippets before implementing into the main programme

#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import Functions as fct


def toolheight(img_left,img_right):
   warped_left,framew_left,frameh_left=fct.warp_img(img_left,150,1,False)
   warped_right,framew_right,fdrameh_left=fct.warp_img(img_right,150,1,False)
  
   cropped_left,toolw_left,toolh_left,x_left,y_left=fct.crop_image(warped_left)
   cropped_right,toolw_right,toolh_right,x_right,y_right=fct.crop_image(warped_right)
  
   #print(f'Left: width:{framew_left}, height:{frameh_left}')
   #print(f'Right: width:{framew_right}, height:{framew_right}')
   print(f'dx:{x_right-x_left}, dy:{y_right-y_left}')
   print(f'squared sum:{(x_right-x_left)**2+(y_right-y_left)**2}')

   # cv2.imshow('warped left',warped_left)
   # cv2.imshow('warped right',warped_right)
   # cv2.waitKey(0)

   # cv2.imshow('cropped left',cropped_left)
   # cv2.imshow('cropped right',cropped_right)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
   # key=cv2.waitKey(0)
   # if key==ord('q'):
   #    quit()
   

   

if __name__ == '__main__':
   mtx_right=np.load('./CalData/mtx_right.npy')
   dist_right=np.load('./CalData/dist_right.npy')
   newcameramtx_right=np.load('./CalData/newcameramtx_right.npy')
   mtx_left=np.load('./CalData/mtx_left.npy')
   dist_left=np.load('./CalData/dist_left.npy')
   newcameramtx_left=np.load('./CalData/newcameramtx_left.npy')

   str_left=('/home/pi/Desktop/Testcontours/InbusPolyLeft.jpg','/home/pi/Desktop/Testcontours/handleNoApprLeft.jpg','/home/pi/Desktop/Testcontours/PlumberPlierPolyLeft.jpg','/home/pi/Desktop/Testcontours/cuttingPlierPolyLeft.jpg','/home/pi/Desktop/Testcontours/ScissorNoApprLeft.jpg')
   str_right=('/home/pi/Desktop/Testcontours/InbusPolyRight.jpg','/home/pi/Desktop/Testcontours/handleNoApprRight.jpg','/home/pi/Desktop/Testcontours/PlumberPlierPolyRight.jpg','/home/pi/Desktop/Testcontours/cuttingPlierPolyRight.jpg','/home/pi/Desktop/Testcontours/ScissorNoApprRight.jpg')
   for i in range(5):
      img_left=cv2.imread(str_left[i])
      img_right=cv2.imread(str_right[i])
      img_left=cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
      img_right=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)

      img_left=cv2.undistort(img_left,mtx_left,dist_left,None,newcameramtx_left)
      img_right=cv2.undistort(img_right,mtx_right,dist_right,None,newcameramtx_right)
      print(i)
      toolheight(img_left,img_right)