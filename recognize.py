import cv2, time
import numpy as np
from Vision import Vision
from Controller import Controller

#######   training part    ############### 
# samples = np.loadtxt('generalsamples_seed.data',np.float32)
# responses = np.loadtxt('generalresponses_seed.data',np.float32)
samples = np.loadtxt('generalsamples_divinity.data',np.float32)
responses = np.loadtxt('generalresponses_divinity.data',np.float32)
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
            if  h > 10:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                result += chr(int((results[0][0])))
                # r = chr(int((results[0][0])))
                # if r == '-':
                #     sign = -1
                # elif r == '+':
                #     pass
                # else:
                #     result = r    
                #cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
    # print(result)
    return result
    # print('result: ' + result)
    # try:
    #     return int(result) * sign
    # except ValueError:
    #     return None

# im = cv2.imread('assets/m1.png')

def seed(n):
    v = Vision()
    i = 0
    save_coord = (1180, 1020)
    refine_coord = (1320, 1020)
    c = Controller()

    while i < n:
        min_magic_atk = recognize(v, (670, 1180, 20, 20))
        max_magic_atk = recognize(v, (695, 1180, 20, 20))
        accuracy = recognize(v, (720, 1180, 20, 20))
        if '+' in min_magic_atk and '+' in max_magic_atk and '+' in accuracy:
            print('saving')
            c.move_mouse(save_coord)
            c.left_mouse_click()
        else:
            print('refining')
            c.move_mouse(refine_coord)
            c.left_mouse_click()
        i += 1
        time.sleep(1/2)

def divinity(n):
    v = Vision()
    i = 0
    save_coord = (1575, 1000)
    refine_coord = (1780, 1000)
    c = Controller()

    while i < n:
        result = ''
        atk = recognize(v, (600, 1670, 20, 20))
        p_def = recognize(v, (625, 1670, 20, 20))
        m_def = recognize(v, (655, 1670, 20, 20))
        max_hp = recognize(v, (685, 1670, 20, 20))
        accuracy = recognize(v, (715, 1670, 20, 20))
        evade = recognize(v, (745, 1670, 20, 20))
        result += atk + p_def + m_def + max_hp + accuracy + evade

        if '-' not in result and len(result) > 0:
            print('saving')
            c.move_mouse(save_coord)
            c.left_mouse_click()
        else:
            print('refining')
            c.move_mouse(refine_coord)
            c.left_mouse_click()
        i += 1
        time.sleep(1/2)

seed(400)
# divinity(378)

#cv2.imshow('im',im)
#cv2.imshow('out',out)
# cv2.waitKey(0)
