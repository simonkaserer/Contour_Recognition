
import numpy as np
import cv2
from matplotlib import pyplot as plt

def calc_angle_deg(x1,y1,x2,y2):
   diff_x=x2-x1
   diff_y=y2-y1
   
   #if diff_y==0: diff_y=0.0000001

   return np.arctan2(diff_x,diff_y)*180/np.pi

def smooth_contour(contour,every_nth_point=1,printflag=False):

   # img=cv2.imread('/home/pi/Desktop/Screenshots/CuttingPliersPolyCropped.jpg')
   # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   # corners = cv2.goodFeaturesToTrack(gray,200,0.1,50)
   # corners = np.int0(corners)
   # for i in corners:
   #    x,y = i.ravel()
   #    cv2.circle(img,(x,y),3,255,-1)
   # plt.imshow(img),plt.show()
   
 
   cnt=np.int0(contour)
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
   cv2.imshow('test',cv2.flip(cnt_img,0))
   cv2.waitKey(0)
   cv2.destroyAllWindows

   direction=np.zeros((number_of_points,6))

     
   for i in range(number_of_points):
   
      x=tool_contour[i][0][0]
      y=tool_contour[i][0][1]
      

      x2=tool_contour[i+2][0][0]
      y2=tool_contour[i+2][0][1]
      x3=tool_contour[i+5][0][0]
      y3=tool_contour[i+5][0][1]
      x4=tool_contour[i+10][0][0]
      y4=tool_contour[i+10][0][1]
      direction[i][0]=x
      direction[i][1]=y
      direction[i][2]=np.arctan((x)/(y))*180/np.pi
      direction[i][3]=calc_angle_deg(x,y,x2,y2)
      direction[i][4]=calc_angle_deg(x,y,x3,y3)
      direction[i][5]=calc_angle_deg(x,y,x4,y4)
      
   print(direction)
   # if printflag:
   #    i=0
   #    kinks=[0]
     

       
   #    print(f'kinks: {kinks}')
   #    print(f'nth point: {nth_point}')
   #    print(f'points: {len(tool_contour)}')
   #    print(f'perimeter:{cv2.arcLength(cnt,True)}')



      
    
    
if __name__ == '__main__':   

  file='/home/pi/Desktop/TO1_Cnt.npy'

  contour=np.load(file)
  smooth_contour(contour)
