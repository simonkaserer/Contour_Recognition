import cv2
import depthai as dai
import numpy as np
import os
import ezdxf as dxf

def extraction_polyDP(img,factor_epsilon,threshold_value,border_offset,printsize,printpoints):
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
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
            warped_image=warped_image[0+border_offset:width-border_offset,0+border_offset:height-border_offset]
            # Look for the contour of the tool:
            
            # Look for the contour of the tool:
            cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
            if len(cnts)>0:
                cnt=sorted(cnts,key=cv2.contourArea)[-1]
                # Turn the tool
                (x,y),(w,h),a=cv2.minAreaRect(cnt)
                rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
                rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
                # Crop the tool
                cnts,hierarchy=cv2.findContours(rotated_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnt=sorted(cnts,key=cv2.contourArea)[-1]
                x,y,w,h=cv2.boundingRect(cnt)
               
                cropped_image=rotated_image[y-2:y+h+2,x-2:x+w+2]
                cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                if len(cnts)>0:
                    cnt=sorted(cnts,key=cv2.contourArea)[-1]
                    epsilon=factor_epsilon*cv2.arcLength(cnt,True)
                    tool_contour=cv2.approxPolyDP(cnt,epsilon,True)

                
                    if printpoints:
                        print("Number of appr. points:")
                        print(len(tool_contour))
                    

                    inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
                    cont_tool=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
                    cv2.imshow("tool-curve",cont_tool)
                    return tool_contour
            else:
                print("Tool not found")
        else:
            print("Outer square not found!")
            

def extraction_convexHull(img,threshold_value,border_offset):
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
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

            input_pts=np.float32([pt_A,pt_B,pt_C,pt_D])
            output_pts=np.float32([[0,0],[0,height],[width,height],[width,0]])
            transf_matrix=cv2.getPerspectiveTransform(input_pts,output_pts,)
            warped_image=cv2.warpPerspective(thresh,transf_matrix,(width,height),flags=cv2.INTER_LINEAR)
                # crop the image to remove the outer edge (offset can maybe be smaller when camera calibration is done?)
            warped_image=warped_image[0+border_offset:width-border_offset,0+border_offset:height-border_offset]

            # Look for the contour of the tool:
            cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            if len(cnts)>0:
                cnt=sorted(cnts,key=cv2.contourArea)[-1]
                
                tool_hull=cv2.convexHull(cnt)
                
                inv=cv2.cvtColor(warped_image,cv2.COLOR_GRAY2BGR)
                cont_tool=cv2.drawContours(inv,[tool_hull],-1,(0,255,0),3)

                cv2.imshow("tool-curve",cv2.resize(cont_tool,(600,600)))
                return tool_hull 
            else:
                print("Tool not found")
        else:
            print("Outer square not found!")
    

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
    msp.add_lwpolyline(points)
    file.saveas(path_and_name)



    