from json import tool
import cv2
import depthai as dai
import numpy as np
import os
import ezdxf as dxf

def warp_img(img,threshold_value:int,border_offset_px:int,show_outer_edge:bool): 
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
            warped_image=cv2.warpPerspective(thresh,transf_matrix,(width,height),flags=cv2.INTER_NEAREST)
            #maybe use :
            #warped_image=cv2.warpPerspective(thresh,transf_matrix,(width,height),flags=cv2.INTER_LANCZOS4)
            # or
            #warped_image=cv2.warpPerspective(thresh,transf_matrix,(width,height),flags=cv2.INTER_AREA)

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
        epsilon=factor_epsilon*cv2.arcLength(cnt,True)
        cont=cv2.approxPolyDP(cnt,epsilon,True)
        cont=cont.tolist()
        cont=cont[::every_nth_point]
        tool_contour=np.array(cont)
        #inv=cv2.cvtColor(cropped_image,cv2.COLOR_GRAY2BGR)
        # create a black background
        inv=np.zeros((int(toolheight+2),int(toolwidth+2),3),dtype='uint8')
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
        inv=np.zeros((int(toolheight+2),int(toolwidth+2),3),dtype='uint8')
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
        inv=np.zeros((int(toolheight+2),int(toolwidth+2),3),dtype='uint8')
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


def dxf_exporter(contour,path_and_name,scaling_width,scaling_height): 
    file=dxf.new('R2000')
    msp=file.modelspace()
    points=[]
    #convert the contour to a list:
    cnt=contour.tolist()
    #add the first entry of the contour to the end for a closed contour in dxf
    cnt.append(cnt[0])
    for point in cnt:
        points.append((point[0][0]/scaling_width,point[0][1]/scaling_height))
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

def toolheight(img_left,img_right):
   warped_left,framew_left,frameh_left=warp_img(img_left,150,1,False)
   warped_right,framew_right,fdrameh_left=warp_img(img_right,150,1,False)
  
   cropped_left,toolw_left,toolh_left,x_left,y_left=crop_image(warped_left)
   cropped_right,toolw_right,toolh_right,x_right,y_right=crop_image(warped_right)
  
   print(f'dx:{x_right-x_left}, dy:{y_right-y_left}')
   print(f'squared sum:{(x_right-x_left)**2+(y_right-y_left)**2}')

def rotate_img(img,x_angle,y_angle,z_angle):
   #From yt: v=bbSFk4vTXSI
   if x_angle<0:
      x_angle=360+x_angle
   if y_angle<0:
      y_angle=360+y_angle
   if z_angle<0:
      z_angle=360+z_angle
   w=img.shape[1]
   h=img.shape[0]
   f=(np.sqrt(h**2+w**2))/(2*np.sin(z_angle*np.pi/np.float32(180)) if np.sin(z_angle*np.pi/np.float32(180)) != 0 else 0.000001)
   
   RX=np.array([[1,0,0,0],
                [0,np.cos((x_angle*np.pi/np.float32(180))),-np.sin((x_angle*np.pi/np.float32(180))),0],
                [0,np.sin((x_angle*np.pi/np.float32(180))),np.cos((x_angle*np.pi/np.float32(180))),0],
                [0,0,0,1]])
   RY=np.array([[np.cos((y_angle*np.pi/np.float32(180))),0,-np.sin((y_angle*np.pi/np.float32(180))),0],
                [0,1,0,0],
                [np.sin((y_angle*np.pi/np.float32(180))),0,np.cos((y_angle*np.pi/np.float32(180))),0],
                [0,0,0,1]])
   RZ=np.array([[np.cos((z_angle*np.pi/np.float32(180))),-np.sin((z_angle*np.pi/np.float32(180))),0,0],
                [np.sin((z_angle*np.pi/np.float32(180))),np.cos((z_angle*np.pi/np.float32(180))),0,0],
                [0,0,1,0],
                [0,0,0,1]]) 
                                  
   # Calculate the rotation matrix
   R=np.dot(np.dot(RX,RY),RZ)

   # Translation matrix
   T=np.array([[1,0,0,max(w,h)/3],
               [0,1,0,-max(w,h)/2],
               [0,0,1,f],
               [0,0,0,1]])

   # Projection of 2D matrix -> 3D matrix
   A1=np.array([[1,0,-w/2],
                [0,1,-h/2],
                [0,0,1],
                [0,0,1]])   

   # Projection of 3D matrix -> 2D matrix                
   A2=np.array([[f,0,w,0],
                [0,f,h,0],
                [0,0,1,0]])                            
   M=np.dot(A2,np.dot(T,np.dot(R,A1)))

   img_rot=cv2.warpPerspective(img,M,(max(w,h),max(w,h)),flags=cv2.INTER_NEAREST)

   return img_rot   