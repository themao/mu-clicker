from Controller import Controller
from Vision import Vision

v = Vision()
(img, sct_img) = v.take_screenshot()
v.save(sct_img, 'assets/screenshot.png')
c = Controller()
#c.move_mouse(1370,1070)
#c.left_mouse_click()