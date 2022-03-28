#!/usr/bin/env python3
from itertools import combinations_with_replacement
import sys
import cv2
import numpy as np
import Functions as fct


def toolheight(img_left,img_right):

   cnts_left,_=cv2.findContours(img_left,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
   cnts_right,_=cv2.findContours(img_right,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
   img_left=cv2.cvtColor(img_left,cv2.COLOR_GRAY2BGR)
   img_right=cv2.cvtColor(img_right,cv2.COLOR_GRAY2BGR)
   
    # Sort the contour by area
   cntsL=sorted(cnts_left,key=cv2.contourArea)
   cntsR=sorted(cnts_right,key=cv2.contourArea)
   # choose the frame contour
   frameL=cntsL[-1]
   frameR=cntsR[-1]
   # Calculate the average disparity
   # disp_frame=0.0
   # for ptL,ptR in zip(frameL,frameR):
   #    disp_frame+=ptL[0]-ptR[0]
   # disp_frame/=len(frameL)
   meanL=np.mean(frameL,0)
   meanR=np.mean(frameR,0)
   disp_frame=abs(meanL-meanR)[0]
   print(disp_frame)
   # Choose the tool contour - the last two contours are the illuminated frame 
   toolL=cntsL[-3]
   toolR=cntsR[-3]
   # disp_tool=0.0
   # for ptL,ptR in zip(toolL,toolR):
   #    disp_tool+=ptL[0]-ptR[0]
   # disp_tool/=len(toolL)
   mean_tool_left=np.mean(toolL,0)
   mean_tool_right=np.mean(toolR,0)
   disp_tool=abs(mean_tool_left-mean_tool_right)[0]
   print(disp_tool)
   
   
  
   height_frame=882.5*75/disp_frame[0]
   height_tool=height_frame-(882.5*75/disp_tool[0])
   
   print(f'height tool: {height_tool}')
      
   contsleft=cv2.drawContours(img_left,toolL,-1,(255,0,0),2)
   contsright=cv2.drawContours(img_right,toolR,-1,(255,0,0),2)
   contsleft=cv2.drawContours(contsleft,frameL,-1,(0,0,255),2)
   contsright=cv2.drawContours(contsright,frameR,-1,(0,0,255),2)
   stack_conts=np.hstack((cv2.resize(contsleft,(400,300)),cv2.resize(contsright,(400,300))))
   cv2.imshow("contours",stack_conts)
   key=cv2.waitKey(0)
   if key==ord('q'):
      quit()
   cv2.destroyAllWindows()
   return height_tool
      




if __name__ == '__main__':
   mtx_right=np.load('./CalData/mtx_right.npy')
   dist_right=np.load('./CalData/dist_right.npy')
   newcameramtx_right=np.load('./CalData/newcameramtx_right.npy')
   mtx_left=np.load('./CalData/mtx_left.npy')
   dist_left=np.load('./CalData/dist_left.npy')
   newcameramtx_left=np.load('./CalData/newcameramtx_left.npy')
   stereoMapL_x=np.load('./CalData/stereoMapL_x.npy')
   stereoMapL_y=np.load('./CalData/stereoMapL_y.npy')
   stereoMapR_x=np.load('./CalData/stereoMapR_x.npy')
   stereoMapR_y=np.load('./CalData/stereoMapR_y.npy')

   str_left=('/home/pi/Desktop/Testcontours/InbusPolyLeft.jpg','/home/pi/Desktop/Testcontours/handleNoApprLeft.jpg','/home/pi/Desktop/Testcontours/PlumberPlierPolyLeft.jpg','/home/pi/Desktop/Testcontours/cuttingPlierPolyLeft.jpg','/home/pi/Desktop/Testcontours/ScissorNoApprLeft.jpg')
   str_right=('/home/pi/Desktop/Testcontours/InbusPolyRight.jpg','/home/pi/Desktop/Testcontours/handleNoApprRight.jpg','/home/pi/Desktop/Testcontours/PlumberPlierPolyRight.jpg','/home/pi/Desktop/Testcontours/cuttingPlierPolyRight.jpg','/home/pi/Desktop/Testcontours/ScissorNoApprRight.jpg')
   i=3 # 0 - 4
   img_left=cv2.imread(str_left[i])
   img_right=cv2.imread(str_right[i])
   img_left=cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
   img_right=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)

   

   img_left=cv2.remap(img_left,stereoMapL_x,stereoMapL_y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
   img_right=cv2.remap(img_right,stereoMapR_x,stereoMapR_y,cv2.INTER_LANCZOS4,cv2.BORDER_CONSTANT,0)
   
   _,img_left=cv2.threshold(img_left,100,255,0)
   _,img_right=cv2.threshold(img_right,100,255,0)

   toolheight(img_left,img_right)