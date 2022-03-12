from json import tool
import cv2
import depthai as dai
import numpy as np
import os
import ezdxf as dxf

def warp_img(img,threshold_value,border_offset_px,show_outer_edge): 
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
            
            return warped_image,width,height
            
        else:   
            return None,None,None
    else:
        return None,None,None

def crop_image(warped_image):
    # Look for the contour of the tool:
    cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        # Turn the tool
        (x,y),(w,h),a=cv2.minAreaRect(cnt)
        rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
        rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
        # Crop the tool
        cropped_image=rotated_image[int((y-h/2)-2):int((y+h/2)+2),int((x-w/2)-2):int((x+w/2)+2)]
        return cropped_image,w+4,h+4,x,y
    else:
        return None,None,None,None,None

def extraction_polyDP(cropped_image,factor_epsilon:float,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int):
    cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        cont=cnt.tolist()
        cont=cont[::every_nth_point]
        cnt_nth=np.array(cont)
        epsilon=factor_epsilon*cv2.arcLength(cnt_nth,True)
        tool_contour=cv2.approxPolyDP(cnt,epsilon,True)
        #inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
        # create a black background
        inv=np.zeros((int(toolheight+10),int(toolwidth+10),3),dtype='uint8')
        if connectpoints:
            img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
        else:
            img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),2) #only points
        
        # Scale the image to the width or height of the Contour View window
        height,width,_=img_cont.shape
        if width > height:
            scale=500/width
            
        else:
            scale=500/height
        img_cont_scaled=cv2.resize(img_cont,(int(width*scale),int(height*scale)))
        
        return tool_contour,img_cont_scaled
    else:
        return None,None

def extraction_TehChin(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int):
    # Find the rotated and cropped tool contour
    cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        cont=cnt.tolist()
        cont=cont[::every_nth_point]
        tool_contour=np.array(cont)
       
        # create a black background
        inv=np.zeros((int(toolheight+10),int(toolwidth+10),3),dtype='uint8')
        if connectpoints:
            img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
        else:
            img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),2) #only points
        
        # Scale the image to the width or height of the Contour View window
        height,width,_=img_cont.shape
        if width > height:
            scale=500/width
            
        else:
            scale=500/height
        img_cont_scaled=cv2.resize(img_cont,(int(width*scale),int(height*scale)))
        
        return tool_contour,img_cont_scaled
    else:
        return None,None

def extraction_convexHull(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int): 
    # Look for the contour of the cropped and turned tool:
    cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        cont=cnt.tolist()
        cont=cont[::every_nth_point]
        cnt_nth=np.array(cont)
        # Find the convex hull
        tool_hull=cv2.convexHull(cnt_nth)
        # Draw the contour
        inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
        if connectpoints:
            img_cont=cv2.drawContours(inv,[tool_hull],-1,(0,255,0),2)
        else:
            img_cont=cv2.drawContours(inv,tool_hull,-1,(0,255,0),2) #only points
        # Scale the image to the width or height of the Contour View window
        height,width,_=img_cont.shape
        if width > height:
            scale=500/width
            
        else:
            scale=500/height
        img_cont_scaled=cv2.resize(img_cont,(int(width*scale),int(height*scale)))
        
        return tool_hull,img_cont_scaled
    else:
        return None,None

def extraction_None(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int):
    # Find the rotated and cropped tool contour
    cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        cont=cnt.tolist()
        cont=cont[::every_nth_point]
        tool_contour=np.array(cont)
        # create a black background
        inv=np.zeros((int(toolheight+10),int(toolwidth+10),3),dtype='uint8')
        if connectpoints:
            img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
        else:
            img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),2) #only points
        # Scale the image to the width or height of the Contour View window
        height,width,_=img_cont.shape
        if width > height:
            scale=500/width
            
        else:
            scale=500/height
        img_cont_scaled=cv2.resize(img_cont,(int(width*scale),int(height*scale)))
        
        return tool_contour,img_cont_scaled
    else:
        return None,None


def dxf_exporter(contour,path_and_name,scaling): 
    file=dxf.new('R2000')
    msp=file.modelspace()
    points=[]
    #convert the contour to a list:
    cnt=contour.tolist()
    #add the first entry of the contour to the end for a closed contour in dxf
    cnt.append(cnt[0])
    for point in cnt:
        points.append((point[0][0]/scaling,point[0][1]/scaling))
    msp.add_lwpolyline(points)
    file.saveas(path_and_name)

def dxf_exporter_spline(contour,path_and_name,scaling): 
    # Spline is not working with InkScape - Fusion 360 works (Inventor should also)
    file=dxf.new('R2000')
    msp=file.modelspace()
    points=[]
    #convert the contour to a list:
    cnt=contour.tolist()
    #add the first entry of the contour to the end for a closed contour in dxf
    cnt.append(cnt[0])
    for point in cnt:
        points.append((point[0][0]/scaling,point[0][1]/scaling,0))
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
