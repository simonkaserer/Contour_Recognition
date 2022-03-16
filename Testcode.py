# This file is for testing code snippets before implementing into the main programme

#!/usr/bin/env python3
from json import tool
import sys
import cv2
import numpy as np
import Functions as fct


def smooth_contour(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int,printflag):
   
   cnts,_=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   if len(cnts)>0:
      cnt=max(cnts,key=cv2.contourArea)
      cont=cnt.tolist()
      cont=cont[::every_nth_point]
      tool_contour=np.array(cont)
      
    
      nth_point=round(len(cnt)*3/cv2.arcLength(cnt,True),0)

      
         

      # create a black background
      inv=np.zeros((int(toolheight+2),int(toolwidth+2),3),dtype='uint8')
      if connectpoints:
         img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),1)
      else:
         img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),1) #only points
      if printflag:
         for point in tool_contour:
            x=point[0][0]
            y=point[0][1]
            x2=(point+1)[0][0]
            y2=(point+1)[0][1]
            x3=(point+2)[0][0]
            y3=(point+3)[0][1]
            direction0=np.arctan((x)/(y))
            direction1=np.arctan((x2-x)/(y2-y))
            direction2=np.arctan((x3-x)/(y3-y))
            cv2.circle(img_cont,(x,y),10,(150,30,255),-1)
            print(f'x:{x}, y:{y}')
            print(f'x2:{x2}, y2:{y2}')
            print(f'direction0:{direction0}')
            print(f'direction1:{direction1}')
            print(f'direction2:{direction2}')
            key=cv2.waitKey(0)
            cv2.imshow('points',img_cont)
            if key==ord('q'):
               break
         print(f'nth point: {nth_point}')
         print(f'points: {len(tool_contour)}')
         print(f'perimeter:{cv2.arcLength(cnt,True)}')
      
      
      
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
   
  
   

   

if __name__ == '__main__':
  
   
   str=('/home/pi/Desktop/Testcontours/InbusPoly.jpg','/home/pi/Desktop/Testcontours/handleNoAppr.jpg','/home/pi/Desktop/Testcontours/PlumberPlierPoly.jpg','/home/pi/Desktop/Testcontours/cuttingPlierPoly.jpg','/home/pi/Desktop/Testcontours/ScissorNoAppr.jpg')
   printflag=False
   connectpoints=True
   points=1
   img=cv2.imread(str[3])
   
   w=img.shape[1]
   h=img.shape[0]
   img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   img=cv2.Canny(img,230,240,L2gradient=False)
 
   while True:
      cnt,cnt_img=smooth_contour(img,points,connectpoints,w,h,printflag)
      printflag=False
      cv2.imshow('contour',cnt_img)
      key=cv2.waitKey(100)
      if key==ord('q'):
         cv2.destroyAllWindows
         quit()
      if key==ord('c'):
         connectpoints = not connectpoints
      if key==ord('+'):
         points+=1
         print(f'npoints:{points}')
      if key==ord('-') and points >1:
         points-=1
         print(f'npoints:{points}')
      if key==ord('p'):
         printflag=True