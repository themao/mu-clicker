import cv2
from mss import mss
from mss import tools
from PIL import Image
import numpy as np
import time

class Vision:
    def __init__(self):
        self.static_templates = {
            'left-goalpost': 'assets/left-goalpost.png',
            'bison-head': 'assets/bison-head.png',
            'pineapple-head': 'assets/pineapple-head.png',
            'bison-health-bar': 'assets/bison-health-bar.png',
            'pineapple-health-bar': 'assets/pineapple-health-bar.png',
            'cancel-button': 'assets/cancel-button.png',
            'filled-with-goodies': 'assets/filled-with-goodies.png',
            'next-button': 'assets/next-button.png',
            'tap-to-continue': 'assets/tap-to-continue.png',
            'unlocked': 'assets/unlocked.png',
            'full-rocket': 'assets/full-rocket.png'
        }

        self.templates = { k: cv2.imread(v, 0) for (k, v) in self.static_templates.items() }

        self.monitor = {'top': 0, 'left': 0, 'width': 2560, 'height': 1440}
        self.screen = mss()

        self.frame = None
    
    def convert_rgb_to_bgr(self, img):
        return img[:, :, ::-1]

    def take_screenshot(self, params=None):
        if params and len(params) == 4:
            self.monitor = {'top': params[0], 'left': params[1], 'width': params[2], 'height': params[3]}

        sct_img = self.screen.grab(self.monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        img = np.array(img)
        img = self.convert_rgb_to_bgr(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return (img_gray, sct_img)

    def save(self, img, output):
        tools.to_png(img.rgb, img.size, output=output)

    def refresh_frame(self):
        self.frame = self.take_screenshot()
    
    def match_template(self, img_grayscale, template, threshold=0.9):
        """
        Matches template image in a target grayscaled image
        """

        res = cv2.matchTemplate(img_grayscale, template, cv2.TM_CCOEFF_NORMED)
        matches = np.where(res >= threshold)
        return matches

    def find_template(self, name, image=None, threshold=0.9):
        if image is None:
            if self.frame is None:
                self.refresh_frame()

            image = self.frame

        return self.match_template(
            image,
            self.templates[name],
            threshold
        )