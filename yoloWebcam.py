# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 23:10:07 2018

@author: HatemZam
"""

import numpy as np
import cv2
from darkflow.net.build import TFNet

options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'threshold': 0.15,
        'gpu': 1.0
        }

tfnet = TFNet(options)

colors = [tuple(255*np.random.rand(3)) for i in range(10)]

#cap = cv2.VideoCapture('videoplayback.mp4')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

while(1):
    ret,frame = cap.read()
    #print(frame.shape)
    results = tfnet.return_predict(frame)
    
    if ret:
        
        for c, res in zip(colors, results):
            tl = (res['topleft']['x'], res['topleft']['y'])
            br = (res['bottomright']['x'], res['bottomright']['y'])
            label = res['label']
            frame = cv2.rectangle(frame, tl, br, c, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        
        cv2.imshow('frame',frame)
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cap.release()
            break
        
    else:
        cap.release()
        break

cv2.destroyAllWindows()




