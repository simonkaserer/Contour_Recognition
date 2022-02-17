# This file is for testing code snippets before implementing into the main programme

#!/usr/bin/env python3

import cv2
import numpy as np
import os


#open a picture from the desktop:
path= '/home/pi/Desktop/'
picture = cv2.imread('/home/pi/Desktop/testLeftkey.jpg')

gray=cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,200,255,0)
#cv2.imshow("thresh",thresh)
#cv2.waitKey(0)
#blur = cv2.medianBlur(gray,3)
#cv2.imshow("blur",blur)
#cv2.waitKey(0)
blur=thresh

 
contours,hierarchy = cv2.findContours(blur,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("Contours 1: ")
print(len(contours))
pic=cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
cont1=cv2.drawContours(pic,contours,0,(0,255,0),1)

cv2.imshow('cont1',cont1)

cv2.waitKey(0)

#find max area contour
if len(contours) > 0:
    cnt = sorted(contours,key=cv2.contourArea)[-1]
    (x,y),(w,h),a=cv2.minAreaRect(cnt)
    print("Turning angle:")
    print(a)
    rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
    rotated_image=cv2.warpAffine(blur,rot_mat,(int(w+x),int(h+y)))
    #points=cv2.boxPoints(cnt)

    cv2.imshow("rotated",rotated_image)
    cv2.waitKey(0)
    # Find the max-area contour of the outer line:
    cnts=cv2.findContours(rotated_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts)>0:
        cnt=sorted(cnts,key=cv2.contourArea)[-1]
        epsilon=0.1*cv2.arcLength(cnt,True)
        # use the Douglas-Peucker algorithm for approximating a rectangle shape
        approx=cv2.approxPolyDP(cnt,epsilon,True)
        #find a straight bounding rectangle inside the approximated one
        x,y,w,h=cv2.boundingRect(approx)
        rotated_image=rotated_image[y:y+h,x:x+w].copy()
        cv2.imshow("cropped",rotated_image)
        cv2.waitKey(0)


#print("Contours 2: ")
#print(len(contours))

cont2=cv2.drawContours(pic,contours,1,(0,255,0),1)
#cv2.imwrite(os.path.join(path,'Contour1.jpg'),cont1)
#cv2.imwrite(os.path.join(path,'Contour2.jpg'),cont2)

