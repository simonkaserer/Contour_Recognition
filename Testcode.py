# This file is for testing code snippets before implementing into the main programme

#!/usr/bin/env python3

import cv2
import numpy as np
import os
import ezdxf as dxf

def dxf_exporter(contour,path_and_name):
    file=dxf.new('R2000')
    msp=file.modelspace()
    points=[]
    #convert the contour to a list:
    cnt=contour.tolist()
    #add the first entry of the contour to the end for a closed contour in dxf
    cnt.append(cnt[0])
    for point in cnt:
        points.append((point[0][0],point[0][1]))
        print(point)
    msp.add_lwpolyline(points)
    file.saveas(path_and_name)

#open a picture from the desktop:
path= '/home/pi/Desktop/'
picture = cv2.imread('/home/pi/Desktop/testRightkey.jpg')


gray=cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,160,255,0)
 
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

# Find the max-area contour of the outer line:
if len(contours) > 0:
    cnt = sorted(contours,key=cv2.contourArea)[-1]
    # warp the contour into a straight rectangle:
        # find the cornerpoints of the square
    epsilon=0.01*cv2.arcLength(cnt,True)
        # use the Douglas-Peucker algorithm for approximating a rectangle shape
    outer_square=cv2.approxPolyDP(cnt,epsilon,True)
        # save the points in seperate variables
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
        epsilon=0.0005*cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,epsilon,True)
        print("Number of appr. points:")
        print(len(approx))
        

        inv=cv2.cvtColor(warped_image,cv2.COLOR_GRAY2BGR)
        cont_tool=cv2.drawContours(inv,[approx],-1,(0,255,0),3)

        cv2.imshow("tool-curve",cv2.resize(cont_tool,(600,600)))
        cv2.waitKey(0)


        dxf_exporter(approx,path+'test.dxf')






