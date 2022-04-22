
import numpy as np
import cv2

def smooth_contour(contour):

   
 
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

  file='/home/pi/Desktop/TO1_Cnt.npy'

  contour=np.load(file)
  smooth_contour(contour)
