import cv2
import numpy as np
from Vision import Vision

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

def recognize(v, coord):
    (img, sct_img) = v.take_screenshot(coord)
    cv2.imshow('i', img)
    im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    result = ''
    sign = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10 and area < 200:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if  h > 3:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(15,20))
                roismall = roismall.reshape((1,300))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                r = chr(int((results[0][0])))
                if r == '-':
                    sign = -1
                elif r == '+':
                    pass
                else:
                    result = r    
                #cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
    print('result: ' + result)
    try:
        return int(result) * sign
    except ValueError:
        return None

# im = cv2.imread('assets/m1.png')
v = Vision()
# min_magic_atk = recognize(v, (721, 1176, 28, 20))
max_magic_atk = recognize(v, (745, 1178, 25, 20))
# min_magic_atk = recognize(v, (447, 1765, 20, 15))
print(max_magic_atk)
#cv2.imshow('im',im)
#cv2.imshow('out',out)
cv2.waitKey(0)