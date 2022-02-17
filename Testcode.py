# This file is for testing code snippets before implementing into the main programme

#!/usr/bin/env python3

import cv2
import numpy as np
import os


#open a picture from the desktop:
path= '/home/pi/Desktop/'
picture = cv2.imread('/home/pi/Desktop/testLeftkey.jpg')

# vlt. einen geradheitsabgleich machen ?! zum entzerren des randes

# und evtl canny alg. laufen lassen




gray=cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,200,255,0)
#cv2.imshow("thresh",thresh)
#cv2.waitKey(0)

# no blur!
#blur = cv2.medianBlur(gray,3)
#cv2.imshow("blur",blur)
#cv2.waitKey(0)
blur=thresh

# maybe helpful
#edges=cv2.Canny(blur,100,200)
#print(len(edges))
#cv2.imshow("canny",edges)
#cv2.waitKey(0)
#cv2.imwrite(os.path.join(path,'Canny.jpg'),edges)
 
contours,hierarchy = cv2.findContours(blur,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print("Contours 1: ")
#print(len(contours))
pic=cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)

# Visualize the first contour
#cont1=cv2.drawContours(pic,contours,0,(0,255,0),1)
#cv2.imshow('cont1',cont1)
#cv2.waitKey(0)


#find max area contour
if len(contours) > 0:
    cnt = sorted(contours,key=cv2.contourArea)[-1]
    (x,y),(w,h),a=cv2.minAreaRect(cnt)
    print("Turning angle:")
    print(a)
    rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
    rotated_image=cv2.warpAffine(blur,rot_mat,(int(w+x),int(h+y)))
    #points=cv2.boxPoints(cnt)

    #cv2.imshow("rotated",rotated_image)
    #cv2.waitKey(0)

    # Find the max-area contour of the outer line:
    cnts,hierarchy=cv2.findContours(rotated_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print("contours rotated")
    #print(len(cnts))
    if len(cnts)>0:
        cnt=sorted(cnts,key=cv2.contourArea)[-1]
        epsilon=0.1*cv2.arcLength(cnt,True)
        # use the Douglas-Peucker algorithm for approximating a rectangle shape
        approx=cv2.approxPolyDP(cnt,epsilon,True)
        #find a straight bounding rectangle inside the approximated one
        x,y,w,h=cv2.boundingRect(approx)
        #print("Bounding rect:")
        #print(x)
        #print(y)
        #print(w)
        #print(h)
        # rotated_image=rotated_image[y:y+h,x:x+w].copy()
        # some pixel get cut away to reach the inner side of the contour
        offset=12
        # this offset can be smaller if the outer contour is straight and only 1 px 
        print("width")
        print(w-(2*offset))
        print("height")
        print(h-(2*offset))
        cropped_image=rotated_image[y+offset:y+h-offset,x+offset:x+w-offset]
        #cv2.imshow("cropped",cropped_image)
        #cv2.waitKey(0)

        

        cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        #cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
        #for testing:
        pic=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
        testcont=cv2.drawContours(pic,cnts,-1,(0,255,0),3)
        print("Number of contour points:")
        print(len(cnts))
        cv2.imshow("testcontour",testcont)
        cv2.waitKey(0)
        if len(cnts)>0:
            cnt=sorted(cnts,key=cv2.contourArea)[-1]
            epsilon=0.0005*cv2.arcLength(cnt,True)
            #   use the Douglas-Peucker algorithm for approximating a rectangle shape
            approx=cv2.approxPolyDP(cnt,epsilon,True)
            print("Number of appr. points:")
            print(len(approx))
            

            inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
            cont_tool=cv2.drawContours(inv,[approx],-1,(0,255,0),3)

            cv2.imshow("tool-curve",cont_tool)
            cv2.waitKey(0)






#print("Contours 2: ")
#print(len(contours))

#cont2=cv2.drawContours(pic,contours,1,(0,255,0),1)
#cv2.imwrite(os.path.join(path,'Contour1.jpg'),cont1)
#cv2.imwrite(os.path.join(path,'Contour2.jpg'),cont2)
