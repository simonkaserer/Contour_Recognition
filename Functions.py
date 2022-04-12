import cv2
import numpy as np
import ezdxf as dxf
from scipy.interpolate import interp1d

def warp_img(img,threshold_value:int,border_offset_px:int,show_outer_edge:bool): # This function finds the corners of the lamp perimeter and warps it to a straight image
    # Threshold the image to binarize it
    _,thresh=cv2.threshold(img,threshold_value,255,0)
    # Find the contours in the image to select the biggest one
    contours,_ = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Find the max-area contour of the outer line:
    if len(contours) > 0:
        cnt=max(contours,key=cv2.contourArea)
        #Show the found contour if the parameter is True. This is implemented for test puroses
        if show_outer_edge:
            pic=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
            edge_out=cv2.drawContours(pic,[cnt],-1,(255,0,0),2)
            cv2.imshow("outer edge",cv2.resize(edge_out,(720,500)))
        # warp the contour into a straight rectangle:
            # find the cornerpoints of the square by setting the epsilon to a relatively high value to only get the cornerpoints
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
            # Find the maximum width and height
            width=max(width1,width2)
            height=max(height1,height2)
            # Arrange the points in an array
            input_pts=np.float32([pt_A,pt_B,pt_C,pt_D])
            # Set up the point array of the destination picture
            output_pts=np.float32([[0,0],[0,height],[width,height],[width,0]])
            transf_matrix=cv2.getPerspectiveTransform(input_pts,output_pts,)
            # Warp the image using the warpPerspective function with the Area interpolation flag
            warped_image=cv2.warpPerspective(thresh,transf_matrix,(width,height),flags=cv2.INTER_AREA)
            # crop the image to remove the outer edge
            warped_image=warped_image[0+border_offset_px:width-border_offset_px,0+border_offset_px:height-border_offset_px]
            # Return the warped image along with the framsize and the points of the input images
            return warped_image,width,height,input_pts
        else:   
            return None,None,None,None
    else:
        return None,None,None,None

def crop_image(warped_image): # Search a tool contour and crop the image by using the minAreaRect function of OpenCV
    # Look for the contour of the tool:
    cnts,hierarchy=cv2.findContours(warped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        # Find the minimum enclosing rectangle
        (x,y),(w,h),a=cv2.minAreaRect(cnt)
        # Get the rotation matrix to rotate the tool
        rot_mat=cv2.getRotationMatrix2D((x,y),a,1)
        # Rotate the tool with respect to the angle
        rotated_image=cv2.warpAffine(warped_image,rot_mat,(int(w+x),int(h+y)))
        # Crop the tool
        cropped_image=rotated_image[int((y-h/2)-2):int((y+h/2)+2),int((x-w/2)-2):int((x+w/2)+2)]
        # Return the cropped image along with the size and the position
        return cropped_image,w+4,h+4,x,y
    else:
        return None,None,None,None,None

def extraction_polyDP(cropped_image,factor_epsilon:float,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int):
    # This function extracts the contour with the Douglas Peucker Algorithm
    cnts,_=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # Find the biggest area contour
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        # Set the epsilon for the DP approximation - the lower the epsilon, the more accurate the contour is
        epsilon=factor_epsilon*cv2.arcLength(cnt,True)
        # Find the contour of the tool
        cont=cv2.approxPolyDP(cnt,epsilon,True)
        # Decimate the points if selected via the slider
        cont=cont.tolist()
        cont=cont[::every_nth_point]
        tool_contour=np.array(cont)
        # create a black background
        inv=np.zeros((int(toolheight),int(toolwidth),3),dtype='uint8')
        # Draw the contour
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
        # Return the found contour points and image
        return tool_contour,img_cont_scaled
    else:
        return None,None

def extraction_TehChin(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int):
    # Extracts the contour with the Teh Chin approximation
    cnts,_=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    if len(cnts)>0:
        # Find the maximum area contour
        cnt=max(cnts,key=cv2.contourArea)
        # Convert the points to a list to decimate it
        cont=cnt.tolist()
        cont=cont[::every_nth_point]
        tool_contour=np.array(cont)
        # create a black background
        inv=np.zeros((int(toolheight),int(toolwidth),3),dtype='uint8')
        # Draw the contour
        if connectpoints:
            img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),2)
        else:
            img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),2) 
        # Scale the image to the width or height of the Contour View window
        height,width,_=img_cont.shape
        if width > height:
            scale=500/width
        else:
            scale=500/height
        img_cont_scaled=cv2.resize(img_cont,(int(width*scale),int(height*scale)))
        # Return the scaled image and the contour points
        return tool_contour,img_cont_scaled
    else:
        return None,None

def extraction_convexHull(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int): 
    # This function creates the hull of the tool
    cnts,_=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        # Find the maximum area contour
        cnt=max(cnts,key=cv2.contourArea)
        # Decimate the points according to the slider
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
        # Return the hull and the hull image
        return tool_hull,img_cont_scaled
    else:
        return None,None

def extraction_None(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int):
    # Find the rotated and cropped tool contour with no chain approximation 
    cnts,hierarchy=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        cont=cnt.tolist()
        # Decimate the points if set at the slider
        cont=cont[::every_nth_point]
        tool_contour=np.array(cont)
        # create a black background
        inv=np.zeros((int(toolheight+2),int(toolwidth+2),3),dtype='uint8')
        # Draw the points connected or unconnected according to the checkbox parameter
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
        # Return the contour points and the image
        return tool_contour,img_cont_scaled
    else:
        return None,None

def extraction_spline(cropped_image,every_nth_point:int,toolwidth:int,toolheight:int):
    # Extracts the contour and approximates it with a spline
    cnts,_=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(cnts)>0:
        cnt=max(cnts,key=cv2.contourArea)
        cont=cnt.tolist()
        # The points get decimized by the number every nth point
        cont=cont[::every_nth_point]
        number_of_points=len(cont)
        tool_contour=np.array(cont)
        # Rearrange the points
        points=np.array([[tool_contour[i][0][0],tool_contour[i][0][1]] for i in range(number_of_points)])
        # Transpose the points
        x,y=points.T
        # Set up the vector for spline fitting    
        i=np.arange(0,len(points))
        # Creates a evenly spaced vector with 2 points in between each given point
        interp_i=np.linspace(0,i.max(),2*i.max())
        # Interpolate the contour points
        x_new=interp1d(i,x,kind='cubic')(interp_i)
        y_new=interp1d(i,y,kind='cubic')(interp_i)
        # Rearrange the points to fit the np.array format
        cnt=np.array([[[int(x_new[i]),int(y_new[i])]for i in range(len(x_new))]])
        # create a black background
        inv=np.zeros((int(toolheight),int(toolwidth),3),dtype='uint8')
        # Draw the contour
        img_cont=cv2.drawContours(inv,[cnt],-1,(0,255,0),2)
        # Scale the image to the width or height of the Contour View window
        height,width,_=img_cont.shape
        if width > height:
                scale=500/width
        else:
                scale=500/height
        img_cont_scaled=cv2.resize(img_cont,(int(width*scale),int(height*scale)))
        # Return the contour points and the image
        return tool_contour,img_cont_scaled 
    else:
        return None,None

def extraction_spline_tehChin(cropped_image,every_nth_point:int,toolwidth:int,toolheight:int):
   # Extracts the contour with the Teh Chin approximation and then approximates it with a spline
   cnts,_=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
   if len(cnts)>0:
      cnt=max(cnts,key=cv2.contourArea)
      cont=cnt.tolist()
      cont=cont[::every_nth_point]
      number_of_points=len(cont)
      tool_contour=np.array(cont)
      #Rearrange the points
      points=np.array([[tool_contour[i][0][0],tool_contour[i][0][1]] for i in range(number_of_points)])
      # Transpose the points
      x,y=points.T
      # Create the vector for interpolating
      i=np.arange(0,len(points))
      # Create a evenly spreaded vector with 2 points per step
      interp_i=np.linspace(0,i.max(),2*i.max())
      # Interpolate the points
      x_new=interp1d(i,x,kind='cubic')(interp_i)
      y_new=interp1d(i,y,kind='cubic')(interp_i)
      # Rearrange the points in np.array format
      cnt=np.array([[[int(x_new[i]),int(y_new[i])]for i in range(len(x_new))]])
      # create a black background
      inv=np.zeros((int(toolheight),int(toolwidth),3),dtype='uint8')
      # Draw the contour
      img_cont=cv2.drawContours(inv,[cnt],-1,(0,255,0),2)
      # Scale the image to the width or height of the Contour View window
      height,width,_=img_cont.shape
      if width > height:
            scale=500/width
      else:
            scale=500/height
      img_cont_scaled=cv2.resize(img_cont,(int(width*scale),int(height*scale)))
      # Return the scaled image and the contour points
      return tool_contour,img_cont_scaled 
   else:
        return None,None

def dxf_exporter(contour,path_and_name:str,scaling_framewidth,scaling_frameheight,thickness:int,prefs): # Exports the contour points and scales it
    # Calculate the scaling factor
    if prefs['use_heightscaling']:
        factor_width= 1+prefs['scaling_width']*(1-(thickness/100))
        factor_height= 1+prefs['scaling_height']*(1-(thickness/100))
    else:
        factor_width= 1+prefs['scaling_width']
        factor_height= 1+prefs['scaling_height']
    # Create the file for ezdxf
    file=dxf.new('R2000')
    msp=file.modelspace()
    points=[]
    #convert the contour to a list:
    cnt=contour.tolist()
    #add the first entry of the contour to the end for a closed contour in dxf
    cnt.append(cnt[0])
    for point in cnt:
        points.append((point[0][0]/scaling_framewidth*factor_width,point[0][1]/scaling_frameheight*factor_height))
    # Add the points to the dxf geometry section
    msp.add_lwpolyline(points)
    # Save the file under the filename with absolute path
    file.saveas(path_and_name)

def toolthickness(img_left,img_right,threshold:int): # This function calculates the thickness of the tool
    # The images get binarized with the passed threshold value:
    _,img_left=cv2.threshold(img_left,threshold,255,0)
    _,img_right=cv2.threshold(img_right,threshold,255,0)
    # The contours are captured 
    cnts_left,_=cv2.findContours(img_left,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    cnts_right,_=cv2.findContours(img_right,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    
    # Sort the contours by area
    cntsL=sorted(cnts_left,key=cv2.contourArea)
    cntsR=sorted(cnts_right,key=cv2.contourArea)
    # Thehe frame contours are chosen. The outer contour has the biggest area
    frameL=cntsL[-1]
    frameR=cntsR[-1]
    # Calculate the average disparity of the frame
    disp_frame=0.0
    meanL=np.mean(frameL,0)
    meanR=np.mean(frameR,0)
    disp_frame=abs(meanL-meanR)[0][0]
    # Choose the tool contour - the biggest two contours are the illuminated frames inside and outside contour
    toolL=cntsL[-3]
    toolR=cntsR[-3]
    # Calculate the moments for the tool contours
    ML=cv2.moments(toolL)
    MR=cv2.moments(toolR)
    # Find the centroids of the tools by divideing the spatial moment m10 by m00 which is the area
    xL=ML['m10']/ML['m00']
    xR=MR['m10']/MR['m00']
    # Calculate the value of the disparity of the tool
    disp_tool=abs(xL-xR)    
    # And convert the disparity from pixel to mm with the formulas:
        # With 800P mono camera resolution where HFOV=71.9 degrees
        # focal_length_in_pixels = 1280 * 0.5 / tan(71.9 * 0.5 * PI / 180) = 882.5
    # and
        # For OAK-D @ 800P mono cameras and disparity of eg. 10 pixels
        # depth = 882.5 * 75 / 10 = 6618.8 # mm
    depth_frame=882.5*75/disp_frame
    height_tool=depth_frame-(882.5*75/disp_tool)
    # Round the thickness and convert it to an integer
    height_tool=int(round(height_tool,0))
    # If the integer is smaller than 1 the value given back is set to 0:
    if height_tool <1:
        height_tool=0
    return height_tool
