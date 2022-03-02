from json import tool
import cv2
import depthai as dai
import numpy as np
import os
import ezdxf as dxf

def extraction_polyDP(img,factor_epsilon,threshold_value,border_offset_px,every_nth_point,printsize,printpoints,show_outer_edge):
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt=max(contours,key=cv2.contourArea)
        #Test: Show the found contour:
        if show_outer_edge:
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
            warped_image=warped_image[0+border_offset_px:width-border_offset_px,0+border_offset_px:height-border_offset_px]
            
            # Look for the contour of the tool:
            cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if len(cnts)>0:
                cnt=max(cnts,key=cv2.contourArea)
                # Turn the tool
                (x,y),(w,h),a=cv2.minAreaRect(cnt)
                rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
                rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
               # Crop the tool
                cropped_image=rotated_image[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]
                
                # Find the rotated and cropped tool contour
                cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                if len(cnts)>0:
                    cnt=max(cnts,key=cv2.contourArea)
                    for point in [cnt][::every_nth_point]:
                        cnt_nth=point
                    epsilon=factor_epsilon*cv2.arcLength(cnt_nth,True)
                    tool_contour=cv2.approxPolyDP(cnt,epsilon,True)
                    #inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
                    # create a black background
                    inv=np.zeros((int(h),int(w),3),dtype='uint8')
                    img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
                    if printpoints:
                        print("Number of appr. points:")
                        print(len(tool_contour))
                     #cv2.imshow("tool-curve",img_cont)
                    return tool_contour,img_cont
                else:
                    return None,None
            else:
                return None,None
        else:
            return None,None
    else:
        return None,None

def extraction_TehChin(img,factor_epsilon,threshold_value,border_offset_px,every_nth_point,printsize,printpoints,show_outer_edge):
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt=max(contours,key=cv2.contourArea)
        #Test: Show the found contour:
        if show_outer_edge:
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
            warped_image=warped_image[0+border_offset_px:width-border_offset_px,0+border_offset_px:height-border_offset_px]
            
            # Look for the contour of the tool:
            cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if len(cnts)>0:
                cnt=max(cnts,key=cv2.contourArea)
                # Turn the tool
                (x,y),(w,h),a=cv2.minAreaRect(cnt)
                rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
                rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
               # Crop the tool
                cropped_image=rotated_image[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]
                
                # Find the rotated and cropped tool contour
                cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
                if len(cnts)>0:
                    cnt=max(cnts,key=cv2.contourArea)
                    for point in [cnt][::every_nth_point]:
                        tool_contour=point
                    #tool_contour=cnt
                    if printpoints:
                        print("Number of appr. points:")
                        print(len(tool_contour))
                    #inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
                    # create a black background
                    inv=np.zeros((int(h),int(w),3),dtype='uint8')
                    img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),2)
                     #cv2.imshow("tool-curve",img_cont)
                    return tool_contour,img_cont
                else:
                    return None,None
            else:
                return None,None
        else:
            return None,None
    else:
        return None,None

def extraction_convexHull(img,factor_epsilon,threshold_value,border_offset_px,every_nth_point,printsize,printpoints,show_outer_edge): 
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt=max(contours,key=cv2.contourArea)
        #Test: Show the found contour:
        if show_outer_edge:
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
            warped_image=warped_image[0+border_offset_px:width-border_offset_px,0+border_offset_px:height-border_offset_px]

            # Look for the contour of the tool:
            cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            if len(cnts)>0:
                cnt=max(cnts,key=cv2.contourArea)
                # Turn the tool
                (x,y),(w,h),a=cv2.minAreaRect(cnt)
                rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
                rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
                # Crop the tool
                cropped_image=rotated_image[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]
                # Look for the contour of the cropped and turned tool:
                cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                if len(cnts)>0:
                    cnt=max(cnts,key=cv2.contourArea)
                    for point in [cnt][::every_nth_point]:
                        cnt_nth=point
                    # Find the convex hull
                    tool_hull=cv2.convexHull(cnt_nth)
                    # Draw the contour
                    inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
                    cont_img=cv2.drawContours(inv,[tool_hull],-1,(0,255,0),3)
                    #cv2.imshow("tool-curve",cont_img)
                    return tool_hull,cont_img
                else:
                    return None,None
            else:
                return None,None
        else:
            return None,None
    else:
        return None,None

def extraction_None(img,factor_epsilon,threshold_value,border_offset_px,every_nth_point,printsize,connectpoints,printpoints,show_outer_edge):
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt=max(contours,key=cv2.contourArea)
        #Test: Show the found contour:
        if show_outer_edge:
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
            warped_image=warped_image[0+border_offset_px:width-border_offset_px,0+border_offset_px:height-border_offset_px]
            
            # Look for the contour of the tool:
            cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if len(cnts)>0:
                cnt=max(cnts,key=cv2.contourArea)
                # Turn the tool
                (x,y),(w,h),a=cv2.minAreaRect(cnt)
                rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
                rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
               # Crop the tool
                cropped_image=rotated_image[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]
                
                # Find the rotated and cropped tool contour
                cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                if len(cnts)>0:
                    cnt=max(cnts,key=cv2.contourArea)
                    cont=cnt.tolist()
                    cont=cont[::every_nth_point]
                    tool_contour=np.array(cont)
                    # create a black background
                    inv=np.zeros((int(h),int(w),3),dtype='uint8')
                    if connectpoints:
                        img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
                    else:
                        img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),2) #only points
                    if printpoints:
                        print("Number of appr. points:")
                        print(len(tool_contour))
                     #cv2.imshow("tool-curve",img_cont)
                    return tool_contour,img_cont
                else:
                    return None,None
            else:
                return None,None
        else:
            return None,None
    else:
        return None,None

def extraction_canny(img,factor_epsilon,threshold_value,border_offset_px,every_nth_point,printsize,printpoints,show_outer_edge):
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt=max(contours,key=cv2.contourArea)
        #Test: Show the found contour:
        if show_outer_edge:
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
            warped_image=warped_image[0+border_offset_px:width-border_offset_px,0+border_offset_px:height-border_offset_px]
            
            # Look for the contour of the tool:
            cnts=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if len(cnts)>0:
                cnt=max(cnts,key=cv2.contourArea)
                # Turn the tool
                (x,y),(w,h),a=cv2.minAreaRect(cnt)
                rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
                rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
               # Crop the tool
                cropped_image=rotated_image[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]
                
                # Find the rotated and cropped tool contour
                #cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                cnts=cv2.Canny(cropped_image,100,200)
                if len(cnts)>0:
                    cnt=max(cnts,key=cv2.contourArea)
                    for point in [cnt][::every_nth_point]:
                        tool_contour=point
                    #inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
                    # create a black background
                    inv=np.zeros((int(h),int(w),3),dtype='uint8')
                    img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
                    if printpoints:
                        print("Number of appr. points:")
                        print(len(tool_contour))
                    return tool_contour,img_cont
                else:
                    return None,None
            else:
                return None,None
        else:
            return None,None
    else:
        return None,None


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

def dxf_exporter_spline(contour,path_and_name): 
    # Spline is not working with InkScape - Fusion 360 works (Inventor should also)
    file=dxf.new('R2000')
    msp=file.modelspace()
    points=[]
    #convert the contour to a list:
    cnt=contour.tolist()
    #add the first entry of the contour to the end for a closed contour in dxf
    cnt.append(cnt[0])
    for point in cnt:
        points.append((point[0][0],point[0][1],0))
    msp.add_spline(points)
    file.saveas(path_and_name)

def check_for_square(img,threshold_value):
    ret,thresh=cv2.threshold(img,threshold_value,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt = sorted(contours,key=cv2.contourArea)[-1]
        #Test: Show the found contour:
        pic=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        edge_out=cv2.drawContours(pic,[cnt],-1,(255,0,0),2)
            # find the cornerpoints of the square
        epsilon=0.01*cv2.arcLength(cnt,True)
            # find the square shape and check if it has 4 corners
        outer_square=cv2.approxPolyDP(cnt,epsilon,True)
        if len(outer_square) ==4:
            return True
        else:
            return False
