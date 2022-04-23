
import numpy as np
import cv2

def smooth_contour(contour,every_nth_point=1,printflag=False):

   
 
   cnt=contour
   cont=cnt.tolist()
   cont=cont[::every_nth_point]
   number_of_points=len(cont)
   for i in range(10):
      cont.append(cont[i])
   tool_contour=np.array(cont)


   nth_point=round(len(cnt)*3/cv2.arcLength(cnt,True),0)

   w_vect,h_vect=contour.T
   width=max(max(w_vect))
   height=max(max(h_vect))
   # create a black background
   background=np.zeros((int(height+2),int(width+2),3),dtype='uint8')
   cnt_img=cv2.drawContours(background,contour,-1,(0,255,0),2)
   cv2.imshow('test',cv2.flip(background,0))
   cv2.waitKey(0)
   cv2.destroyAllWindows


     
      
   for i in cont:
      x=tool_contour[i][0][0]
      y=tool_contour[i][0][1]

      x2=tool_contour[i+1][0][0]
      y2=tool_contour[i+1][0][1]
      x3=tool_contour[i+2][0][0]
      y3=tool_contour[i+2][0][1]
      direction0=np.arctan((x)/(y))*180/np.pi
      direction1=np.arctan((x2-x)/(y2-y))*180/np.pi
      direction2=np.arctan((x3-x)/(y3-y))*180/np.pi
      

   if printflag:
      i=0
      kinks=[0]
     

       
      print(f'kinks: {kinks}')
      print(f'nth point: {nth_point}')
      print(f'points: {len(tool_contour)}')
      print(f'perimeter:{cv2.arcLength(cnt,True)}')



      
    
    
if __name__ == '__main__':   

  file='/home/pi/Desktop/TO1_Cnt.npy'

  contour=np.load(file)
  smooth_contour(contour)
