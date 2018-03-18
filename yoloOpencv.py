# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 00:50:54 2018

@author: HatemZam
"""

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
        'threshold': 0.4,
        'gpu': 1.0
        }

tfnet = TFNet(options)

colors = [tuple(255*np.random.rand(3)) for i in range(10)]

img = cv2.imread('sample_img/sample_office.jpg', cv2.IMREAD_COLOR)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (800, 558))

results = tfnet.return_predict(img)
    
for c, res in zip(colors, results):
    tl = (res['topleft']['x'], res['topleft']['y'])
    br = (res['bottomright']['x'], res['bottomright']['y'])
    label = res['label']
    img = cv2.rectangle(img, tl, br, c, 6)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
        
cv2.imshow('frame',img)

while 1:
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()




