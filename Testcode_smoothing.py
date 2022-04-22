
import numpy as np
import cv2




file='/home/pi/Desktop/TO1_Cnt.npy'

contour=np.load(file)
w_vect,h_vect=contour.T
width=max(max(w_vect))
height=max(max(h_vect))
background=np.zeros((int(height+2),int(width+2),3),dtype='uint8')
cnt_img=cv2.drawContours(background,contour,-1,(0,255,0),2)
cv2.imshow('test',cv2.flip(background,0))
cv2.waitKey(0)
cv2.destroyAllWindows