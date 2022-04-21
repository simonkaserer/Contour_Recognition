#!/usr/bin/env python3
from json import tool
from pickletools import float8
import sys
import cv2
import numpy as np
import Functions as fct
#from scipy.interpolate import interp1d

def smooth_contour(cropped_image,every_nth_point:int,connectpoints:bool,toolwidth:int,toolheight:int,printflag,direction):

   cnts,_=cv2.findContours(cropped_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   if len(cnts)>0:
      cnt=max(cnts,key=cv2.contourArea)
      cont=cnt.tolist()
      cont=cont[::every_nth_point]
      number_of_points=len(cont)
      for i in range(10):
         cont.append(cont[i])
      tool_contour=np.array(cont)


      nth_point=round(len(cnt)*3/cv2.arcLength(cnt,True),0)




      # create a black background
      inv=np.zeros((int(toolheight+2),int(toolwidth+2),3),dtype='uint8')
      if connectpoints:
         img_cont=cv2.drawContours(inv,[tool_contour],-1,(0,255,0),1)
      else:
         img_cont=cv2.drawContours(inv,tool_contour,-1,(0,255,0),1) #only points
      for i in range(number_of_points):
         x=tool_contour[i][0][0]
         y=tool_contour[i][0][1]

         x2=tool_contour[i+1][0][0]
         y2=tool_contour[i+1][0][1]
         x3=tool_contour[i+2][0][0]
         y3=tool_contour[i+2][0][1]
         direction0=np.arctan((x)/(y))*180/np.pi
         direction1=np.arctan((x2-x)/(y2-y))*180/np.pi
         direction2=np.arctan((x3-x)/(y3-y))*180/np.pi
         direction[i][0]=direction1
         direction[i][1]=direction2


      if printflag:
         i=0
         kinks=[0]
         while True:
            x=tool_contour[i][0][0]
            y=tool_contour[i][0][1]
            point_img=img_cont.copy()

            direction1=direction[i][0]
            direction2=direction[i][1]
            cv2.circle(point_img,(x,y),2,(150,30,255),-1)

            print(f'direction1:{direction1}')
            print(f'direction2:{direction2}')
            if direction[i][0]==-direction[i+1][0] :
               print('Kink +1')
               kinks.append(i)
            if direction[i][1]==-direction[i+1][1]:
               print('Kink +2')
            if direction[i-1][0]==-direction[i][0]:
               print('kink -1')
            if direction[i-1][1]==-direction[i][1]:
               print('kink -2')
            print('')
            cv2.imshow('points',point_img)
            key=cv2.waitKey(0)
            if key==ord('q'):
               break
            if key==ord('+'):
               i+=1
            if key==ord('-') and i > 0:
               i-=1
         print(f'kinks: {kinks}')
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

      return tool_contour,img_cont_scaled,direction
   else:
        return None,None,None



   
if __name__ == '__main__':
  

   str=('/home/pi/Desktop/Testcontours/InbusPoly.jpg','/home/pi/Desktop/Testcontours/handleNoAppr.jpg','/home/pi/Desktop/Testcontours/PlumberPlierPoly.jpg','/home/pi/Desktop/Testcontours/cuttingPlierPoly.jpg','/home/pi/Desktop/Testcontours/ScissorNoAppr.jpg')
   printflag=False
   connectpoints=False
   points=3
   img=cv2.imread(str[1])

   w=img.shape[1]
   h=img.shape[0]
   img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   img=cv2.Canny(img,230,240,L2gradient=False)
   direction=np.zeros((1000,2))
   while True:
      cnt,cnt_img,directions=smooth_contour(img,points,connectpoints,w,h,printflag,direction)
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
      if key==ord('s'):
         cv2.imwrite('/home/pi/Desktop/points.png',cnt_img)
      if key==ord('d'):
         for direction in directions:
            if direction[0] != 0 and direction[1] != 0:
               print(direction) 
